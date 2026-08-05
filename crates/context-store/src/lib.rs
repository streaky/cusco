use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

pub type Token = i32;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug)]
struct SequenceNode {
    parent: Option<Arc<SequenceNode>>,
    token: Token,
    len: usize,
    hash: BranchId,
}

#[derive(Clone, Debug, Default)]
pub struct PersistentTokenSequence {
    tail: Option<Arc<SequenceNode>>,
}

impl PersistentTokenSequence {
    pub fn len(&self) -> usize {
        self.tail.as_ref().map_or(0, |node| node.len)
    }

    pub fn is_empty(&self) -> bool {
        self.tail.is_none()
    }

    pub fn id(&self) -> BranchId {
        self.tail
            .as_ref()
            .map_or_else(empty_branch_id, |node| node.hash)
    }

    pub fn append(&self, tokens: &[Token]) -> Self {
        let mut tail = self.tail.clone();
        for &token in tokens {
            let (len, parent_hash) = tail
                .as_ref()
                .map_or((1, empty_branch_id()), |node| (node.len + 1, node.hash));
            let hash = branch_hash(parent_hash, token, len);
            tail = Some(Arc::new(SequenceNode {
                parent: tail,
                token,
                len,
                hash,
            }));
        }
        Self { tail }
    }

    pub fn prefix(&self, len: usize) -> Option<Self> {
        if len > self.len() {
            return None;
        }
        let mut tail = self.tail.clone();
        while tail.as_ref().is_some_and(|node| node.len > len) {
            tail = tail.and_then(|node| node.parent.clone());
        }
        Some(Self { tail })
    }

    pub fn tokens(&self) -> Vec<Token> {
        let mut result = Vec::with_capacity(self.len());
        let mut current = self.tail.as_deref();
        while let Some(node) = current {
            result.push(node.token);
            current = node.parent.as_deref();
        }
        result.reverse();
        result
    }

    fn shares_prefix_node(&self, other: &Self, len: usize) -> bool {
        match (self.prefix(len), other.prefix(len)) {
            (Some(left), Some(right)) => match (left.tail, right.tail) {
                (None, None) => true,
                (Some(left), Some(right)) => Arc::ptr_eq(&left, &right),
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
    tokens: Vec<Token>,
}

#[derive(Clone, Debug)]
pub struct LogicalContext {
    pub id: LogicalContextId,
    pub tokens: PersistentTokenSequence,
    pub model_epoch: ModelEpoch,
    pub adapter_epoch: AdapterEpoch,
    pub evaluated: Option<EvaluatedPrefixId>,
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

#[derive(Debug)]
pub struct PreparedPublication {
    context: LogicalContextId,
    expected_current: Option<EvaluatedPrefixId>,
    mapping: EvaluatedPrefix,
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
        context.tokens = context.tokens.append(tokens);
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
        let old = context.evaluated.take();
        context.model_epoch = model_epoch;
        context.adapter_epoch = adapter_epoch;
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
        let tokens = prefix.tokens();
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
        let lineage = dependency_hash(parent_lineage, &tokens);
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
                tokens,
            },
        })
    }

    pub fn commit_publication(
        &mut self,
        prepared: PreparedPublication,
    ) -> Result<Arc<EvaluatedPrefix>, Error> {
        let current = self
            .contexts
            .get(&prepared.context)
            .ok_or(Error::ContextNotFound)?
            .evaluated;
        if current != prepared.expected_current {
            return Err(Error::PublicationConflict);
        }
        let context = self.contexts.get(&prepared.context).unwrap();
        if context.model_epoch != prepared.mapping.model_epoch
            || context.adapter_epoch != prepared.mapping.adapter_epoch
        {
            return Err(Error::IncompatibleEpoch);
        }
        if context
            .tokens
            .prefix(prepared.mapping.represented_end)
            .unwrap()
            .tokens()
            != prepared.mapping.tokens
        {
            return Err(Error::PublicationConflict);
        }
        let id = prepared.mapping.id;
        if let Some(existing) = self.mappings.get(&prepared.mapping.id) {
            if *existing.mapping != prepared.mapping {
                return Err(Error::IdentityConflict);
            }
        } else {
            if let Some(parent) = prepared.mapping.parent {
                self.mappings
                    .get_mut(&parent)
                    .ok_or(Error::InvalidDependency)?
                    .references
                    .dependents += 1;
            }
            self.mappings.insert(
                prepared.mapping.id,
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
            self.contexts.get_mut(&prepared.context).unwrap().evaluated = Some(id);
        }
        Ok(self.mappings[&id].mapping.clone())
    }

    pub fn longest_valid_prefix(
        &self,
        context_id: LogicalContextId,
    ) -> Result<Option<Arc<EvaluatedPrefix>>, Error> {
        let context = self
            .contexts
            .get(&context_id)
            .ok_or(Error::ContextNotFound)?;
        Ok(self
            .mappings
            .values()
            .filter(|entry| self.mapping_valid_for_context(&entry.mapping, context, None))
            .max_by_key(|entry| entry.mapping.represented_end)
            .map(|entry| entry.mapping.clone()))
    }

    pub fn references(&self, id: EvaluatedPrefixId) -> Option<ReferenceCounts> {
        self.mappings.get(&id).map(|entry| entry.references)
    }

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
            || context
                .tokens
                .prefix(mapping.represented_end)
                .unwrap()
                .tokens()
                != mapping.tokens
        {
            return false;
        }
        match mapping.parent {
            None => true,
            Some(parent) => self.mappings.get(&parent).is_some_and(|entry| {
                entry.mapping.represented_end < mapping.represented_end
                    && self.mapping_valid_for_context(
                        &entry.mapping,
                        context,
                        Some(mapping.represented_end),
                    )
            }),
        }
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
        let removable = self.mappings.get(&id).is_some_and(|entry| {
            entry.references.catalog == 0
                && entry.references.contexts == 0
                && entry.references.dependents == 0
        });
        if !removable {
            return;
        }
        let parent = self
            .mappings
            .remove(&id)
            .and_then(|entry| entry.mapping.parent);
        if let Some(parent) = parent {
            if let Some(entry) = self.mappings.get_mut(&parent) {
                entry.references.dependents = entry.references.dependents.saturating_sub(1);
            }
            self.reclaim(parent);
        }
    }
}

fn empty_branch_id() -> BranchId {
    BranchId(Sha256::digest(b"cusco-empty-sequence").into())
}

fn branch_hash(parent: BranchId, token: Token, len: usize) -> BranchId {
    let mut hash = Sha256::new();
    hash.update(b"cusco-sequence-v1");
    hash.update(parent.0);
    hash.update(token.to_le_bytes());
    hash.update(len.to_le_bytes());
    BranchId(hash.finalize().into())
}

fn dependency_hash(parent: DependencyHash, tokens: &[Token]) -> DependencyHash {
    let mut hash = Sha256::new();
    hash.update(b"cusco-dependency-v1");
    hash.update(parent.0);
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
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
    hash.update(b"cusco-evaluated-prefix-v1");
    hash.update(model.0.to_le_bytes());
    hash.update(adapter.0.to_le_bytes());
    hash.update(parent.map_or([0; 32], |id| id.0));
    hash.update(branch.0);
    hash.update(end.to_le_bytes());
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
        assert!(left.shares_prefix_node(&right, 3));
        assert!(!left.shares_prefix_node(&right, 4));
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
        let mapping = store.commit_publication(prepared).unwrap();
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

    proptest! {
        #[test]
        fn arbitrary_branches_preserve_exact_shared_prefix(prefix in prop::collection::vec(any::<i32>(), 0..64), left in prop::collection::vec(any::<i32>(), 0..32), right in prop::collection::vec(any::<i32>(), 0..32)) {
            let base = PersistentTokenSequence::default().append(&prefix);
            let left_branch = base.append(&left);
            let right_branch = base.append(&right);
            prop_assert!(left_branch.shares_prefix_node(&right_branch, prefix.len()));
            prop_assert_eq!(&left_branch.tokens()[..prefix.len()], prefix.as_slice());
            prop_assert_eq!(&right_branch.tokens()[..prefix.len()], prefix.as_slice());
        }
    }
}
