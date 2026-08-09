use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use thiserror::Error;

pub type Token = i32;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        pub struct $name(pub [u8; 32]);
    };
}

id_type!(BranchId);
id_type!(EvaluatedPrefixId);
id_type!(DependencyHash);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct LogicalContextId(pub u64);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ModelEpoch(pub u64);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AdapterEpoch(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComponentMask(u8);

impl ComponentMask {
    pub const EMPTY: Self = Self(0);
    pub const GLOBAL_KV: Self = Self(1);
    pub const SWA: Self = Self(2);
    pub const RECURRENT: Self = Self(4);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
}

/// Maximum number of tokens held by one immutable logical chunk.
///
/// This is deliberately independent of any native KV block size: logical
/// chunks are the structurally shared identity unit, while executor-owned
/// representations describe their own physical coverage.
pub const LOGICAL_CHUNK_TOKENS: usize = 256;

#[derive(Clone, Debug)]
struct SequenceTail {
    chunk: Arc<SequenceChunk>,
    visible: usize,
}

#[derive(Debug)]
struct SequenceChunk {
    parent: Option<SequenceTail>,
    start_len: usize,
    tokens: Box<[Token]>,
    start_hash: Sha256,
    end_hash: BranchId,
}

#[derive(Clone, Debug, Default)]
pub struct PersistentTokenSequence {
    tail: Option<SequenceTail>,
}

impl PersistentTokenSequence {
    pub fn len(&self) -> usize {
        self.tail
            .as_ref()
            .map_or(0, |tail| tail.chunk.start_len + tail.visible)
    }

    pub fn is_empty(&self) -> bool {
        self.tail.is_none()
    }

    pub fn id(&self) -> BranchId {
        let Some(tail) = &self.tail else {
            return empty_branch_id();
        };
        if tail.visible == tail.chunk.tokens.len() {
            tail.chunk.end_hash
        } else {
            let mut hash = tail.chunk.start_hash.clone();
            hash_tokens(&mut hash, &tail.chunk.tokens[..tail.visible]);
            BranchId(hash.finalize().into())
        }
    }

    /// Adds tokens as bounded immutable chunks, allocating once per chunk
    /// rather than once per token.
    pub fn append(&self, tokens: &[Token]) -> Self {
        let mut sequence = self.clone();
        let mut hash = sequence.hash_state();
        for tokens in tokens.chunks(LOGICAL_CHUNK_TOKENS) {
            let start_hash = hash.clone();
            hash_tokens(&mut hash, tokens);
            let chunk = Arc::new(SequenceChunk {
                parent: sequence.tail.clone(),
                start_len: sequence.len(),
                tokens: tokens.into(),
                start_hash,
                end_hash: BranchId(hash.clone().finalize().into()),
            });
            sequence.tail = Some(SequenceTail {
                visible: chunk.tokens.len(),
                chunk,
            });
        }
        sequence
    }

    pub fn prefix(&self, len: usize) -> Option<Self> {
        if len > self.len() {
            return None;
        }
        let mut tail = self.tail.clone();
        loop {
            let Some(current) = tail else {
                return Some(Self::default());
            };
            if len > current.chunk.start_len {
                return Some(Self {
                    tail: Some(SequenceTail {
                        visible: len - current.chunk.start_len,
                        chunk: current.chunk,
                    }),
                });
            }
            tail = current.chunk.parent.clone();
        }
    }

    pub fn tokens(&self) -> Vec<Token> {
        let mut tokens = self.iter_rev().collect::<Vec<_>>();
        tokens.reverse();
        tokens
    }

    fn hash_state(&self) -> Sha256 {
        let Some(tail) = &self.tail else {
            return empty_sequence_hash();
        };
        let mut hash = tail.chunk.start_hash.clone();
        hash_tokens(&mut hash, &tail.chunk.tokens[..tail.visible]);
        hash
    }

    fn iter_rev(&self) -> ReverseTokens<'_> {
        ReverseTokens {
            tail: self.tail.as_ref(),
            offset: self.tail.as_ref().map_or(0, |tail| tail.visible),
        }
    }

    #[cfg(test)]
    fn shares_prefix_storage(&self, other: &Self, len: usize) -> bool {
        match (self.prefix(len), other.prefix(len)) {
            (Some(left), Some(right)) => match (left.tail, right.tail) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    left.visible == right.visible && Arc::ptr_eq(&left.chunk, &right.chunk)
                }
                _ => false,
            },
            _ => false,
        }
    }
}

impl PartialEq for PersistentTokenSequence {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        match (&self.tail, &other.tail) {
            (None, None) => true,
            (Some(left), Some(right))
                if left.visible == right.visible && Arc::ptr_eq(&left.chunk, &right.chunk) =>
            {
                true
            }
            _ => self.iter_rev().eq(other.iter_rev()),
        }
    }
}

impl Eq for PersistentTokenSequence {}

struct ReverseTokens<'a> {
    tail: Option<&'a SequenceTail>,
    offset: usize,
}

impl Iterator for ReverseTokens<'_> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let tail = self.tail?;
            if self.offset > 0 {
                self.offset -= 1;
                return Some(tail.chunk.tokens[self.offset]);
            }
            self.tail = tail.chunk.parent.as_ref();
            self.offset = self.tail.map_or(0, |parent| parent.visible);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluatedPrefix {
    pub id: EvaluatedPrefixId,
    pub branch: BranchId,
    pub represented_end: usize,
    pub model_epoch: ModelEpoch,
    pub adapter_epoch: AdapterEpoch,
    pub parent: Option<EvaluatedPrefixId>,
    pub lineage: DependencyHash,
    pub required_components: ComponentMask,
    pub complete_components: ComponentMask,
    pub evaluation_parameters: [u8; 32],
    sequence: PersistentTokenSequence,
}

#[derive(Clone, Debug)]
pub struct LogicalContext {
    pub id: LogicalContextId,
    pub tokens: PersistentTokenSequence,
    pub model_epoch: ModelEpoch,
    pub adapter_epoch: AdapterEpoch,
    pub evaluated: Option<EvaluatedPrefixId>,
    /// Monotonic semantic revision used to reject stale prepared publications.
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReferenceCounts {
    pub catalog: usize,
    pub contexts: usize,
    pub dependents: usize,
}

#[derive(Debug)]
struct MappingEntry {
    mapping: Arc<EvaluatedPrefix>,
    references: ReferenceCounts,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MappingEpochKey {
    model_epoch: ModelEpoch,
    adapter_epoch: AdapterEpoch,
}

#[derive(Debug)]
pub struct PreparedPublication {
    context: LogicalContextId,
    expected_revision: u64,
    expected_current: Option<EvaluatedPrefixId>,
    mapping: EvaluatedPrefix,
}

impl PreparedPublication {
    pub fn mapping_id(&self) -> EvaluatedPrefixId {
        self.mapping.id
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum Error {
    #[error("logical context does not exist")]
    ContextNotFound,
    #[error("represented prefix exceeds the token sequence")]
    PrefixOutOfBounds,
    #[error("evaluated prefix has incompatible model or adapter epochs")]
    IncompatibleEpoch,
    #[error("evaluated prefix is incomplete")]
    IncompleteComponents,
    #[error("evaluated prefix dependency is invalid")]
    InvalidDependency,
    #[error("context changed after publication was prepared")]
    PublicationConflict,
    #[error("evaluated prefix does not exist")]
    MappingNotFound,
    #[error("evaluated prefix identity conflicts with published content")]
    IdentityConflict,
}

#[derive(Debug, Default)]
pub struct ContextStore {
    next_context: u64,
    contexts: HashMap<LogicalContextId, LogicalContext>,
    mappings: HashMap<EvaluatedPrefixId, MappingEntry>,
    prefix_index: HashMap<MappingEpochKey, BTreeMap<usize, Vec<EvaluatedPrefixId>>>,
}

impl ContextStore {
    pub fn create(
        &mut self,
        tokens: PersistentTokenSequence,
        model_epoch: ModelEpoch,
        adapter_epoch: AdapterEpoch,
    ) -> LogicalContextId {
        let id = LogicalContextId(self.next_context);
        self.next_context += 1;
        self.contexts.insert(
            id,
            LogicalContext {
                id,
                tokens,
                model_epoch,
                adapter_epoch,
                revision: 0,
                evaluated: None,
            },
        );
        id
    }

    pub fn branch(
        &mut self,
        source: LogicalContextId,
        at: usize,
    ) -> Result<LogicalContextId, Error> {
        let source = self.contexts.get(&source).ok_or(Error::ContextNotFound)?;
        let tokens = source.tokens.prefix(at).ok_or(Error::PrefixOutOfBounds)?;
        Ok(self.create(tokens, source.model_epoch, source.adapter_epoch))
    }

    pub fn append(&mut self, context: LogicalContextId, tokens: &[Token]) -> Result<(), Error> {
        let context = self
            .contexts
            .get_mut(&context)
            .ok_or(Error::ContextNotFound)?;
        if !tokens.is_empty() {
            context.tokens = context.tokens.append(tokens);
            context.revision = context
                .revision
                .checked_add(1)
                .expect("logical context revision exhausted");
        }
        Ok(())
    }

    pub fn context(&self, id: LogicalContextId) -> Option<&LogicalContext> {
        self.contexts.get(&id)
    }

    pub fn remove_context(&mut self, id: LogicalContextId) -> Result<(), Error> {
        let context = self.contexts.remove(&id).ok_or(Error::ContextNotFound)?;
        if let Some(mapping) = context.evaluated {
            self.adjust_context_refs(mapping, false);
        }
        Ok(())
    }

    pub fn set_epochs(
        &mut self,
        id: LogicalContextId,
        model_epoch: ModelEpoch,
        adapter_epoch: AdapterEpoch,
    ) -> Result<(), Error> {
        let context = self.contexts.get_mut(&id).ok_or(Error::ContextNotFound)?;
        if context.model_epoch == model_epoch && context.adapter_epoch == adapter_epoch {
            return Ok(());
        }
        let old = context.evaluated.take();
        context.model_epoch = model_epoch;
        context.adapter_epoch = adapter_epoch;
        context.revision = context
            .revision
            .checked_add(1)
            .expect("logical context revision exhausted");
        if let Some(mapping) = old {
            self.adjust_context_refs(mapping, false);
        }
        Ok(())
    }

    pub fn prepare_publication(
        &self,
        context_id: LogicalContextId,
        represented_end: usize,
        parent: Option<EvaluatedPrefixId>,
        required_components: ComponentMask,
        complete_components: ComponentMask,
        evaluation_parameters: [u8; 32],
    ) -> Result<PreparedPublication, Error> {
        let context = self
            .contexts
            .get(&context_id)
            .ok_or(Error::ContextNotFound)?;
        if represented_end > context.tokens.len() {
            return Err(Error::PrefixOutOfBounds);
        }
        if !complete_components.contains(required_components) {
            return Err(Error::IncompleteComponents);
        }
        let prefix = context.tokens.prefix(represented_end).unwrap();
        let parent_lineage = match parent {
            Some(id) => {
                let entry = self.mappings.get(&id).ok_or(Error::InvalidDependency)?;
                if entry.mapping.represented_end >= represented_end
                    || !self.mapping_valid_for_context(
                        &entry.mapping,
                        context,
                        Some(represented_end),
                    )
                {
                    return Err(Error::InvalidDependency);
                }
                entry.mapping.lineage
            }
            None => DependencyHash([0; 32]),
        };
        let lineage = dependency_hash(parent_lineage, prefix.id(), represented_end);
        let id = mapping_id(
            context.model_epoch,
            context.adapter_epoch,
            parent,
            prefix.id(),
            represented_end,
            required_components,
            evaluation_parameters,
            lineage,
        );
        Ok(PreparedPublication {
            context: context_id,
            expected_current: context.evaluated,
            expected_revision: context.revision,
            mapping: EvaluatedPrefix {
                id,
                branch: prefix.id(),
                represented_end,
                model_epoch: context.model_epoch,
                adapter_epoch: context.adapter_epoch,
                parent,
                lineage,
                required_components,
                complete_components,
                evaluation_parameters,
                sequence: prefix,
            },
        })
    }

    /// Validate that a prepared publication can commit without mutating the store.
    pub fn validate_publication(&self, prepared: &PreparedPublication) -> Result<(), Error> {
        let context = self
            .contexts
            .get(&prepared.context)
            .ok_or(Error::ContextNotFound)?;
        if context.evaluated != prepared.expected_current
            || context.revision != prepared.expected_revision
        {
            return Err(Error::PublicationConflict);
        }
        if context.model_epoch != prepared.mapping.model_epoch
            || context.adapter_epoch != prepared.mapping.adapter_epoch
        {
            return Err(Error::IncompatibleEpoch);
        }
        if context
            .tokens
            .prefix(prepared.mapping.represented_end)
            .as_ref()
            != Some(&prepared.mapping.sequence)
        {
            return Err(Error::PublicationConflict);
        }
        if let Some(existing) = self.mappings.get(&prepared.mapping.id) {
            if *existing.mapping != prepared.mapping {
                return Err(Error::IdentityConflict);
            }
        } else if prepared
            .mapping
            .parent
            .is_some_and(|parent| !self.mappings.contains_key(&parent))
        {
            return Err(Error::InvalidDependency);
        }
        Ok(())
    }

    pub fn commit_publication(
        &mut self,
        prepared: PreparedPublication,
    ) -> Result<Arc<EvaluatedPrefix>, Error> {
        self.validate_publication(&prepared)?;
        let context = self.contexts.get(&prepared.context).unwrap();
        let current = context.evaluated;
        let id = prepared.mapping.id;
        if !self.mappings.contains_key(&prepared.mapping.id) {
            if let Some(parent) = prepared.mapping.parent {
                self.mappings
                    .get_mut(&parent)
                    .ok_or(Error::InvalidDependency)?
                    .references
                    .dependents += 1;
            }
            let lookup_key = MappingEpochKey {
                model_epoch: prepared.mapping.model_epoch,
                adapter_epoch: prepared.mapping.adapter_epoch,
            };
            self.prefix_index
                .entry(lookup_key)
                .or_default()
                .entry(prepared.mapping.represented_end)
                .or_default()
                .push(id);
            self.mappings.insert(
                id,
                MappingEntry {
                    mapping: Arc::new(prepared.mapping),
                    references: ReferenceCounts {
                        catalog: 1,
                        contexts: 0,
                        dependents: 0,
                    },
                },
            );
        }

        if current != Some(id) {
            if let Some(old) = current {
                self.adjust_context_refs(old, false);
            }
            self.adjust_context_refs(id, true);
            let context = self.contexts.get_mut(&prepared.context).unwrap();
            context.evaluated = Some(id);
            context.revision = context
                .revision
                .checked_add(1)
                .expect("logical context revision exhausted");
        }
        Ok(self.mappings[&id].mapping.clone())
    }

    /// Returns immutable mapping metadata. The returned `Arc` does not retain
    /// catalog membership or, in later phases, any physical representation.
    pub fn longest_valid_prefix(
        &self,
        context_id: LogicalContextId,
    ) -> Result<Option<Arc<EvaluatedPrefix>>, Error> {
        let context = self
            .contexts
            .get(&context_id)
            .ok_or(Error::ContextNotFound)?;
        let key = MappingEpochKey {
            model_epoch: context.model_epoch,
            adapter_epoch: context.adapter_epoch,
        };
        let Some(lengths) = self.prefix_index.get(&key) else {
            return Ok(None);
        };
        Ok(lengths
            .range(..=context.tokens.len())
            .rev()
            .find_map(|(_, ids)| self.valid_mapping_from_ids(context, ids)))
    }
    /// Returns immutable mapping metadata without conferring residency ownership.
    pub fn mapping(&self, id: EvaluatedPrefixId) -> Option<Arc<EvaluatedPrefix>> {
        self.mappings.get(&id).map(|entry| entry.mapping.clone())
    }

    /// Reports store-owned logical references; external `Arc` clones are
    /// metadata snapshots and intentionally are not included.
    pub fn references(&self, id: EvaluatedPrefixId) -> Option<ReferenceCounts> {
        self.mappings.get(&id).map(|entry| entry.references)
    }

    /// Releases the catalog's ownership reference. Caller-held metadata
    /// snapshots do not keep this mapping discoverable.
    pub fn release_mapping(&mut self, id: EvaluatedPrefixId) -> Result<(), Error> {
        let entry = self.mappings.get_mut(&id).ok_or(Error::MappingNotFound)?;
        entry.references.catalog = entry.references.catalog.saturating_sub(1);
        self.reclaim(id);
        Ok(())
    }

    fn mapping_valid_for_context(
        &self,
        mapping: &EvaluatedPrefix,
        context: &LogicalContext,
        child_end: Option<usize>,
    ) -> bool {
        if mapping.model_epoch != context.model_epoch
            || mapping.adapter_epoch != context.adapter_epoch
            || !mapping
                .complete_components
                .contains(mapping.required_components)
            || mapping.represented_end > context.tokens.len()
            || child_end.is_some_and(|end| mapping.represented_end > end)
            || context.tokens.prefix(mapping.represented_end).as_ref() != Some(&mapping.sequence)
        {
            return false;
        }

        let mut child = mapping;
        while let Some(parent_id) = child.parent {
            let Some(parent) = self.mappings.get(&parent_id).map(|entry| &*entry.mapping) else {
                return false;
            };
            if parent.represented_end >= child.represented_end
                || parent.model_epoch != child.model_epoch
                || parent.adapter_epoch != child.adapter_epoch
                || !parent
                    .complete_components
                    .contains(parent.required_components)
            {
                return false;
            }
            child = parent;
        }
        true
    }

    fn valid_mapping_from_ids(
        &self,
        context: &LogicalContext,
        ids: &[EvaluatedPrefixId],
    ) -> Option<Arc<EvaluatedPrefix>> {
        ids.iter()
            .filter_map(|id| self.mappings.get(id))
            .filter(|entry| self.mapping_valid_for_context(&entry.mapping, context, None))
            .map(|entry| entry.mapping.clone())
            .min_by_key(|mapping| mapping.id)
    }

    fn adjust_context_refs(&mut self, id: EvaluatedPrefixId, increment: bool) {
        if let Some(entry) = self.mappings.get_mut(&id) {
            if increment {
                entry.references.contexts += 1;
            } else {
                entry.references.contexts = entry.references.contexts.saturating_sub(1);
                self.reclaim(id);
            }
        }
    }

    fn reclaim(&mut self, id: EvaluatedPrefixId) {
        let mut candidate = Some(id);
        while let Some(id) = candidate {
            let removable = self.mappings.get(&id).is_some_and(|entry| {
                entry.references.catalog == 0
                    && entry.references.contexts == 0
                    && entry.references.dependents == 0
            });
            if !removable {
                break;
            }
            let removed = self.mappings.remove(&id).unwrap();
            let key = MappingEpochKey {
                model_epoch: removed.mapping.model_epoch,
                adapter_epoch: removed.mapping.adapter_epoch,
            };
            let mut remove_epoch = false;
            if let Some(lengths) = self.prefix_index.get_mut(&key) {
                let end = removed.mapping.represented_end;
                if let Some(ids) = lengths.get_mut(&end) {
                    ids.retain(|candidate| *candidate != id);
                    if ids.is_empty() {
                        lengths.remove(&end);
                    }
                }
                remove_epoch = lengths.is_empty();
            }
            if remove_epoch {
                self.prefix_index.remove(&key);
            }
            candidate = removed.mapping.parent;
            if let Some(parent) = candidate {
                if let Some(entry) = self.mappings.get_mut(&parent) {
                    entry.references.dependents = entry.references.dependents.saturating_sub(1);
                }
            }
        }
    }
}

fn empty_sequence_hash() -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"cusco-sequence-v2");
    hash
}

fn empty_branch_id() -> BranchId {
    BranchId(empty_sequence_hash().finalize().into())
}

fn hash_tokens(hash: &mut Sha256, tokens: &[Token]) {
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
}

fn dependency_hash(
    parent: DependencyHash,
    branch: BranchId,
    represented_end: usize,
) -> DependencyHash {
    let mut hash = Sha256::new();
    hash.update(b"cusco-dependency-v2");
    hash.update(parent.0);
    hash.update(branch.0);
    hash.update(
        u64::try_from(represented_end)
            .expect("represented prefix length exceeds the portable identity format")
            .to_le_bytes(),
    );
    DependencyHash(hash.finalize().into())
}

#[allow(clippy::too_many_arguments)]
fn mapping_id(
    model: ModelEpoch,
    adapter: AdapterEpoch,
    parent: Option<EvaluatedPrefixId>,
    branch: BranchId,
    end: usize,
    components: ComponentMask,
    parameters: [u8; 32],
    lineage: DependencyHash,
) -> EvaluatedPrefixId {
    let mut hash = Sha256::new();
    hash.update(b"cusco-evaluated-prefix-v2");
    hash.update(model.0.to_le_bytes());
    hash.update(adapter.0.to_le_bytes());
    hash.update(parent.map_or([0; 32], |id| id.0));
    hash.update(branch.0);
    hash.update(
        u64::try_from(end)
            .expect("evaluated prefix length exceeds the portable identity format")
            .to_le_bytes(),
    );
    hash.update([components.0]);
    hash.update(parameters);
    hash.update(lineage.0);
    EvaluatedPrefixId(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn components() -> ComponentMask {
        ComponentMask::GLOBAL_KV
            .union(ComponentMask::SWA)
            .union(ComponentMask::RECURRENT)
    }

    fn publish(
        store: &mut ContextStore,
        context: LogicalContextId,
        end: usize,
        parent: Option<EvaluatedPrefixId>,
    ) -> Arc<EvaluatedPrefix> {
        let prepared = store
            .prepare_publication(context, end, parent, components(), components(), [7; 32])
            .unwrap();
        store.commit_publication(prepared).unwrap()
    }

    #[test]
    fn branches_share_structure_without_sharing_private_tails() {
        let base = PersistentTokenSequence::default().append(&[1, 2, 3]);
        let left = base.append(&[4]);
        let right = base.append(&[5]);
        assert!(left.shares_prefix_storage(&right, 3));
        assert!(!left.shares_prefix_storage(&right, 4));
        assert_eq!(left.tokens(), [1, 2, 3, 4]);
        assert_eq!(right.tokens(), [1, 2, 3, 5]);
    }

    #[test]
    fn publication_is_atomic_and_epoch_changes_invalidate_mappings() {
        let mut store = ContextStore::default();
        let id = store.create(
            PersistentTokenSequence::default().append(&[1, 2, 3]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        let prepared = store
            .prepare_publication(id, 2, None, components(), components(), [7; 32])
            .unwrap();
        assert!(store.longest_valid_prefix(id).unwrap().is_none());
        store.append(id, &[4]).unwrap();
        assert_eq!(
            store.commit_publication(prepared).unwrap_err(),
            Error::PublicationConflict
        );
        let mapping = publish(&mut store, id, 2, None);
        assert_eq!(
            store.longest_valid_prefix(id).unwrap().unwrap().id,
            mapping.id
        );
        store
            .set_epochs(id, ModelEpoch(2), AdapterEpoch(1))
            .unwrap();
        assert!(store.longest_valid_prefix(id).unwrap().is_none());
    }

    #[test]
    fn concurrent_publications_cannot_replace_the_active_mapping() {
        let mut store = ContextStore::default();
        let id = store.create(
            PersistentTokenSequence::default().append(&[1, 2, 3]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        let first = store
            .prepare_publication(id, 1, None, components(), components(), [1; 32])
            .unwrap();
        let stale = store
            .prepare_publication(id, 2, None, components(), components(), [2; 32])
            .unwrap();
        let committed = store.commit_publication(first).unwrap();
        assert_eq!(
            store.commit_publication(stale).unwrap_err(),
            Error::PublicationConflict
        );
        assert_eq!(store.context(id).unwrap().evaluated, Some(committed.id));
    }

    #[test]
    fn publication_revision_rejects_aba_binding_changes() {
        let mut store = ContextStore::default();
        let id = store.create(
            PersistentTokenSequence::default().append(&[1, 2, 3]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        let first = store
            .prepare_publication(id, 1, None, components(), components(), [1; 32])
            .unwrap();
        let first = store.commit_publication(first).unwrap();
        let stale = store
            .prepare_publication(id, 3, None, components(), components(), [3; 32])
            .unwrap();
        let second = store
            .prepare_publication(id, 2, None, components(), components(), [2; 32])
            .unwrap();
        store.commit_publication(second).unwrap();
        let restore_first = store
            .prepare_publication(id, 1, None, components(), components(), [1; 32])
            .unwrap();
        assert_eq!(
            store.commit_publication(restore_first).unwrap().id,
            first.id
        );
        assert_eq!(
            store.commit_publication(stale).unwrap_err(),
            Error::PublicationConflict
        );
    }

    #[test]
    fn longest_lookup_confirms_tokens_after_hash_match() {
        let mut store = ContextStore::default();
        let source = store.create(
            PersistentTokenSequence::default().append(&[1, 2]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        publish(&mut store, source, 2, None);
        let published_tail = store.context(source).unwrap().tokens.tail.clone().unwrap();
        let colliding_chunk = Arc::new(SequenceChunk {
            parent: published_tail.chunk.parent.clone(),
            start_len: published_tail.chunk.start_len,
            tokens: Box::new([1, 99]),
            start_hash: published_tail.chunk.start_hash.clone(),
            end_hash: published_tail.chunk.end_hash,
        });
        let colliding_sequence = PersistentTokenSequence {
            tail: Some(SequenceTail {
                visible: colliding_chunk.tokens.len(),
                chunk: colliding_chunk,
            }),
        };
        let collision = store.create(colliding_sequence, ModelEpoch(1), AdapterEpoch(1));
        assert!(store.longest_valid_prefix(collision).unwrap().is_none());
    }

    #[test]
    fn longest_lookup_validates_tokens_lineage_and_dependencies() {
        let mut store = ContextStore::default();
        let source = store.create(
            PersistentTokenSequence::default().append(&[1, 2, 3, 4]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        let first = publish(&mut store, source, 2, None);
        let second = publish(&mut store, source, 4, Some(first.id));
        let branch = store.branch(source, 2).unwrap();
        store.append(branch, &[9]).unwrap();
        assert_eq!(
            store.longest_valid_prefix(branch).unwrap().unwrap().id,
            first.id
        );
        store.append(branch, &[4]).unwrap();
        assert_eq!(
            store.longest_valid_prefix(branch).unwrap().unwrap().id,
            first.id
        );
        assert_eq!(
            store.longest_valid_prefix(source).unwrap().unwrap().id,
            second.id
        );
    }

    #[test]
    fn reference_accounting_reclaims_dependency_chains() {
        let mut store = ContextStore::default();
        let context = store.create(
            PersistentTokenSequence::default().append(&[1, 2]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        let parent = publish(&mut store, context, 1, None);
        let child = publish(&mut store, context, 2, Some(parent.id));
        assert_eq!(store.references(parent.id).unwrap().dependents, 1);
        assert_eq!(store.references(child.id).unwrap().contexts, 1);
        store.remove_context(context).unwrap();
        store.release_mapping(parent.id).unwrap();
        store.release_mapping(child.id).unwrap();
        assert_eq!(store.references(parent.id), None);
        assert_eq!(store.references(child.id), None);
    }

    #[test]
    fn rejects_incomplete_and_out_of_bounds_publications() {
        let mut store = ContextStore::default();
        let context = store.create(
            PersistentTokenSequence::default().append(&[1]),
            ModelEpoch(1),
            AdapterEpoch(1),
        );
        assert_eq!(
            store
                .prepare_publication(context, 2, None, components(), components(), [0; 32])
                .unwrap_err(),
            Error::PrefixOutOfBounds
        );
        assert_eq!(
            store
                .prepare_publication(
                    context,
                    1,
                    None,
                    components(),
                    ComponentMask::GLOBAL_KV,
                    [0; 32]
                )
                .unwrap_err(),
            Error::IncompleteComponents
        );
        let root = publish(&mut store, context, 1, None);
        assert_eq!(
            store
                .prepare_publication(
                    context,
                    1,
                    Some(root.id),
                    components(),
                    components(),
                    [0; 32]
                )
                .unwrap_err(),
            Error::InvalidDependency
        );
        assert_eq!(
            store.remove_context(LogicalContextId(99)),
            Err(Error::ContextNotFound)
        );
    }

    #[test]
    fn chunking_is_bounded_and_identity_is_append_segmentation_independent() {
        let tokens = (0..(LOGICAL_CHUNK_TOKENS * 3 + 17))
            .map(|token| token as Token)
            .collect::<Vec<_>>();
        let one_append = PersistentTokenSequence::default().append(&tokens);
        let many_appends = tokens
            .chunks(13)
            .fold(PersistentTokenSequence::default(), |sequence, tokens| {
                sequence.append(tokens)
            });

        let mut chunk_count = 0;
        let mut tail = one_append.tail.as_ref();
        while let Some(current) = tail {
            chunk_count += 1;
            assert!(current.chunk.tokens.len() <= LOGICAL_CHUNK_TOKENS);
            tail = current.chunk.parent.as_ref();
        }

        assert_eq!(chunk_count, 4);
        assert_eq!(one_append, many_appends);
        assert_eq!(one_append.id(), many_appends.id());
        assert_eq!(
            one_append.prefix(LOGICAL_CHUNK_TOKENS + 7),
            many_appends.prefix(LOGICAL_CHUNK_TOKENS + 7)
        );
    }

    proptest! {
        #[test]
        fn arbitrary_branches_preserve_exact_shared_prefix(prefix in prop::collection::vec(any::<i32>(), 0..64), left in prop::collection::vec(any::<i32>(), 0..32), right in prop::collection::vec(any::<i32>(), 0..32)) {
            let base = PersistentTokenSequence::default().append(&prefix);
            let left_branch = base.append(&left);
            let right_branch = base.append(&right);
            prop_assert!(left_branch.shares_prefix_storage(&right_branch, prefix.len()));
            prop_assert_eq!(&left_branch.tokens()[..prefix.len()], prefix.as_slice());
            prop_assert_eq!(&right_branch.tokens()[..prefix.len()], prefix.as_slice());
        }
    }
}
