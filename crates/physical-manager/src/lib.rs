use cusco_context_store::{ComponentMask, EvaluatedPrefixId, LogicalContextId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
        pub struct $name(pub u64);
    };
}

id_type!(PhysicalRepresentationId);
id_type!(ActiveBindingId);
id_type!(PreparedTransitionId);
id_type!(TransferId);
id_type!(GrowthReservationId);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum Component {
    GlobalKv,
    SlidingWindow,
    Recurrent,
}

impl Component {
    fn mask(self) -> ComponentMask {
        match self {
            Self::GlobalKv => ComponentMask::GLOBAL_KV,
            Self::SlidingWindow => ComponentMask::SWA,
            Self::Recurrent => ComponentMask::RECURRENT,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Tier {
    Device,
    Host,
    Storage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TransitionClass {
    ReferenceOnly,
    NonDestructive,
    Quiesced,
    Recompute,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReferenceCounts {
    pub logical: usize,
    pub active: usize,
    pub reservations: usize,
    pub transfers: usize,
}

impl ReferenceCounts {
    fn protected(self) -> bool {
        self.active != 0 || self.reservations != 0 || self.transfers != 0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhysicalRepresentation {
    pub id: PhysicalRepresentationId,
    pub mapping: EvaluatedPrefixId,
    pub component: Component,
    pub represented_end: usize,
    pub bytes: usize,
    pub device: bool,
    pub host: bool,
    pub storage: bool,
    pub references: ReferenceCounts,
    pub last_used: u64,
    pub reuse_value: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capacity {
    pub device_bytes: usize,
    pub host_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct Metrics {
    pub device_total: usize,
    pub device_active: usize,
    pub device_growth_reserved: usize,
    pub device_transition_reserved: usize,
    pub device_detached_transfer_reserved: usize,
    pub device_warm: usize,
    pub host_total: usize,
    pub host_used: usize,
    pub transfer_bytes: u64,
    pub promotions: u64,
    pub demotions: u64,
    pub evictions: u64,
    pub prepared: u64,
    pub committed: u64,
    pub aborted: u64,
    pub recomputed: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub enum TraceEvent {
    Registered {
        representation: PhysicalRepresentationId,
        tier: Tier,
        bytes: usize,
    },
    Prepared {
        transition: PreparedTransitionId,
        class: TransitionClass,
        reserved_bytes: usize,
    },
    TransferCompleted {
        transfer: TransferId,
        bytes: usize,
        success: bool,
    },
    Committed {
        transition: PreparedTransitionId,
        binding: ActiveBindingId,
    },
    Aborted {
        transition: PreparedTransitionId,
    },
    Promoted {
        representation: PhysicalRepresentationId,
        bytes: usize,
    },
    Demoted {
        representation: PhysicalRepresentationId,
        bytes: usize,
    },
    Evicted {
        representation: PhysicalRepresentationId,
        tier: Tier,
        bytes: usize,
    },
}

#[derive(Clone, Debug)]
struct ActiveBinding {
    id: ActiveBindingId,
    mapping: EvaluatedPrefixId,
    representations: Vec<PhysicalRepresentationId>,
}

#[derive(Clone, Debug)]
struct PreparedTransition {
    context: LogicalContextId,
    revision: u64,
    mapping: EvaluatedPrefixId,
    representations: Vec<PhysicalRepresentationId>,
    transfers: Vec<TransferId>,
    reserved_bytes: usize,
    class: TransitionClass,
}

#[derive(Clone, Debug)]
struct Transfer {
    representation: PhysicalRepresentationId,
    bytes: usize,
    completed: bool,
    success: bool,
    detached: bool,
}

#[derive(Clone, Copy, Debug)]
struct GrowthReservation {
    bytes: usize,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum Error {
    #[error("physical representation does not exist")]
    RepresentationNotFound,
    #[error("active binding does not exist")]
    BindingNotFound,
    #[error("prepared transition does not exist")]
    TransitionNotFound,
    #[error("transfer does not exist")]
    TransferNotFound,
    #[error("growth reservation does not exist")]
    GrowthReservationNotFound,
    #[error("device capacity is exhausted")]
    DeviceCapacity,
    #[error("host capacity is exhausted")]
    HostCapacity,
    #[error("composite physical state is incomplete or inconsistent")]
    InvalidComposite,
    #[error("no source copy is available for promotion")]
    NoPromotionSource,
    #[error("transition still has incomplete transfers")]
    TransferPending,
    #[error("a transition transfer failed")]
    TransferFailed,
    #[error("logical context revision changed")]
    StaleRevision,
    #[error("physical representation is protected by an owner")]
    Protected,
}

#[derive(Debug)]
pub struct PhysicalManager {
    capacity: Capacity,
    next_id: u64,
    clock: u64,
    representations: HashMap<PhysicalRepresentationId, PhysicalRepresentation>,
    bindings: HashMap<LogicalContextId, ActiveBinding>,
    transitions: HashMap<PreparedTransitionId, PreparedTransition>,
    transfers: HashMap<TransferId, Transfer>,
    growth: HashMap<GrowthReservationId, GrowthReservation>,
    metrics: Metrics,
    events: Vec<TraceEvent>,
}

impl PhysicalManager {
    pub fn new(capacity: Capacity) -> Self {
        Self {
            capacity,
            next_id: 1,
            clock: 0,
            representations: HashMap::new(),
            bindings: HashMap::new(),
            transitions: HashMap::new(),
            transfers: HashMap::new(),
            growth: HashMap::new(),
            metrics: Metrics {
                device_total: capacity.device_bytes,
                host_total: capacity.host_bytes,
                ..Metrics::default()
            },
            events: Vec::new(),
        }
    }

    fn id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("physical manager ID exhausted");
        id
    }

    fn tick(&mut self) -> u64 {
        self.clock = self
            .clock
            .checked_add(1)
            .expect("physical manager clock exhausted");
        self.clock
    }

    pub fn register(
        &mut self,
        mapping: EvaluatedPrefixId,
        component: Component,
        represented_end: usize,
        bytes: usize,
        tier: Tier,
        reuse_value: u64,
    ) -> Result<PhysicalRepresentationId, Error> {
        match tier {
            Tier::Device => self.ensure_device_capacity(bytes)?,
            Tier::Host => self.ensure_host_capacity(bytes)?,
            Tier::Storage => {}
        }
        let id = PhysicalRepresentationId(self.id());
        let last_used = self.tick();
        let mut representation = PhysicalRepresentation {
            id,
            mapping,
            component,
            represented_end,
            bytes,
            device: false,
            host: false,
            storage: false,
            references: ReferenceCounts {
                logical: 1,
                ..ReferenceCounts::default()
            },
            last_used,
            reuse_value,
        };
        match tier {
            Tier::Device => representation.device = true,
            Tier::Host => representation.host = true,
            Tier::Storage => representation.storage = true,
        }
        self.representations.insert(id, representation);
        self.events.push(TraceEvent::Registered {
            representation: id,
            tier,
            bytes,
        });
        self.refresh_capacity_metrics();
        Ok(id)
    }

    pub fn representation(&self, id: PhysicalRepresentationId) -> Option<&PhysicalRepresentation> {
        self.representations.get(&id)
    }

    pub fn active_binding(
        &self,
        context: LogicalContextId,
    ) -> Option<(ActiveBindingId, EvaluatedPrefixId)> {
        self.bindings
            .get(&context)
            .map(|binding| (binding.id, binding.mapping))
    }
    pub fn unbind(&mut self, context: LogicalContextId) -> Result<(), Error> {
        let binding = self
            .bindings
            .remove(&context)
            .ok_or(Error::BindingNotFound)?;
        for representation in binding.representations {
            self.representations
                .get_mut(&representation)
                .unwrap()
                .references
                .active -= 1;
        }
        self.refresh_capacity_metrics();
        Ok(())
    }

    pub fn release_logical_reference(&mut self, id: PhysicalRepresentationId) -> Result<(), Error> {
        let representation = self
            .representations
            .get_mut(&id)
            .ok_or(Error::RepresentationNotFound)?;
        representation.references.logical = representation.references.logical.saturating_sub(1);
        if representation.references.logical == 0 && !representation.references.protected() {
            self.representations.remove(&id);
        }
        self.refresh_capacity_metrics();
        Ok(())
    }

    pub fn reserve_growth(&mut self, bytes: usize) -> Result<GrowthReservationId, Error> {
        self.ensure_device_capacity(bytes)?;
        let id = GrowthReservationId(self.id());
        self.growth.insert(id, GrowthReservation { bytes });
        self.refresh_capacity_metrics();
        Ok(id)
    }

    pub fn release_growth(&mut self, id: GrowthReservationId) -> Result<(), Error> {
        self.growth
            .remove(&id)
            .ok_or(Error::GrowthReservationNotFound)?;
        self.refresh_capacity_metrics();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_transition(
        &mut self,
        context: LogicalContextId,
        revision: u64,
        mapping: EvaluatedPrefixId,
        represented_end: usize,
        required: ComponentMask,
        representations: &[PhysicalRepresentationId],
        allow_recompute: bool,
    ) -> Result<(PreparedTransitionId, TransitionClass, Vec<TransferId>), Error> {
        self.validate_composite(mapping, represented_end, required, representations)?;
        let mut missing_bytes = 0usize;
        let mut recompute = false;
        for id in representations {
            let representation = &self.representations[id];
            if !representation.device {
                if representation.host || representation.storage {
                    missing_bytes = missing_bytes
                        .checked_add(representation.bytes)
                        .ok_or(Error::DeviceCapacity)?;
                } else if allow_recompute {
                    recompute = true;
                    missing_bytes = missing_bytes
                        .checked_add(representation.bytes)
                        .ok_or(Error::DeviceCapacity)?;
                } else {
                    return Err(Error::NoPromotionSource);
                }
            }
        }
        self.ensure_device_capacity(missing_bytes)?;
        let class = if recompute {
            TransitionClass::Recompute
        } else if missing_bytes == 0 {
            TransitionClass::ReferenceOnly
        } else if self.bindings.contains_key(&context) {
            TransitionClass::NonDestructive
        } else {
            TransitionClass::Quiesced
        };
        let transition_id = PreparedTransitionId(self.id());
        let mut transfer_ids = Vec::new();
        for representation in representations {
            let needs_transfer = !self.representations[representation].device
                && (self.representations[representation].host
                    || self.representations[representation].storage);
            self.representations
                .get_mut(representation)
                .unwrap()
                .references
                .reservations += 1;
            if needs_transfer {
                let bytes = self.representations[representation].bytes;
                let transfer = TransferId(self.id());
                self.representations
                    .get_mut(representation)
                    .unwrap()
                    .references
                    .transfers += 1;
                self.transfers.insert(
                    transfer,
                    Transfer {
                        representation: *representation,
                        bytes,
                        completed: false,
                        success: false,
                        detached: false,
                    },
                );
                transfer_ids.push(transfer);
            }
        }
        self.transitions.insert(
            transition_id,
            PreparedTransition {
                context,
                revision,
                mapping,
                representations: representations.to_vec(),
                transfers: transfer_ids.clone(),
                reserved_bytes: missing_bytes,
                class,
            },
        );
        self.metrics.prepared += 1;
        self.events.push(TraceEvent::Prepared {
            transition: transition_id,
            class,
            reserved_bytes: missing_bytes,
        });
        self.refresh_capacity_metrics();
        Ok((transition_id, class, transfer_ids))
    }
    pub fn transition_class(&self, id: PreparedTransitionId) -> Option<TransitionClass> {
        self.transitions.get(&id).map(|transition| transition.class)
    }

    pub fn complete_transfer(&mut self, id: TransferId, success: bool) -> Result<(), Error> {
        let (representation, bytes, already_completed) = {
            let transfer = self.transfers.get(&id).ok_or(Error::TransferNotFound)?;
            (transfer.representation, transfer.bytes, transfer.completed)
        };
        if already_completed {
            return Ok(());
        }
        if success {
            let needs_device = !self.representations[&representation].device;
            if needs_device {
                self.ensure_device_space(bytes)?;
                self.representations
                    .get_mut(&representation)
                    .unwrap()
                    .device = true;
                self.metrics.promotions += 1;
                self.events.push(TraceEvent::Promoted {
                    representation,
                    bytes,
                });
            }
            self.metrics.transfer_bytes += bytes as u64;
        }
        let transfer = self.transfers.get_mut(&id).unwrap();
        transfer.completed = true;
        transfer.success = success;
        let detached = transfer.detached;
        self.representations
            .get_mut(&representation)
            .unwrap()
            .references
            .transfers -= 1;
        self.events.push(TraceEvent::TransferCompleted {
            transfer: id,
            bytes,
            success,
        });
        self.refresh_capacity_metrics();
        if detached {
            self.transfers.remove(&id);
        }
        Ok(())
    }

    pub fn complete_recompute(
        &mut self,
        transition: PreparedTransitionId,
        representation: PhysicalRepresentationId,
    ) -> Result<(), Error> {
        let prepared = self
            .transitions
            .get(&transition)
            .ok_or(Error::TransitionNotFound)?;
        if prepared.class != TransitionClass::Recompute
            || !prepared.representations.contains(&representation)
        {
            return Err(Error::InvalidComposite);
        }
        if !self.representations[&representation].device {
            let bytes = self.representations[&representation].bytes;
            self.ensure_device_space(bytes)?;
            self.representations
                .get_mut(&representation)
                .unwrap()
                .device = true;
            self.metrics.recomputed += 1;
        }
        self.refresh_capacity_metrics();
        Ok(())
    }

    pub fn commit_transition(
        &mut self,
        id: PreparedTransitionId,
        current_revision: u64,
    ) -> Result<ActiveBindingId, Error> {
        let transition = self.transitions.get(&id).ok_or(Error::TransitionNotFound)?;
        if transition.revision != current_revision {
            return Err(Error::StaleRevision);
        }
        for transfer in &transition.transfers {
            let transfer = &self.transfers[transfer];
            if !transfer.completed {
                return Err(Error::TransferPending);
            }
            if !transfer.success {
                return Err(Error::TransferFailed);
            }
        }
        if transition
            .representations
            .iter()
            .any(|rep| !self.representations[rep].device)
        {
            return Err(Error::TransferPending);
        }
        let transition = self.transitions.remove(&id).unwrap();
        self.release_transition_refs(&transition);
        if let Some(old) = self.bindings.remove(&transition.context) {
            for representation in old.representations {
                self.representations
                    .get_mut(&representation)
                    .unwrap()
                    .references
                    .active -= 1;
            }
        }
        for representation in &transition.representations {
            let rep = self.representations.get_mut(representation).unwrap();
            rep.references.active += 1;
            rep.last_used = self.clock;
        }
        let binding = ActiveBindingId(self.id());
        self.bindings.insert(
            transition.context,
            ActiveBinding {
                id: binding,
                mapping: transition.mapping,
                representations: transition.representations,
            },
        );
        self.metrics.committed += 1;
        for transfer in transition.transfers {
            self.transfers.remove(&transfer);
        }
        self.events.push(TraceEvent::Committed {
            transition: id,
            binding,
        });
        self.refresh_capacity_metrics();
        Ok(binding)
    }

    pub fn abort_transition(&mut self, id: PreparedTransitionId) -> Result<(), Error> {
        let mut transition = self
            .transitions
            .remove(&id)
            .ok_or(Error::TransitionNotFound)?;
        for transfer in &transition.transfers {
            if self.transfers[transfer].completed {
                self.transfers.remove(transfer);
            } else {
                self.transfers.get_mut(transfer).unwrap().detached = true;
            }
        }
        transition.transfers.clear();
        self.release_transition_refs(&transition);
        self.metrics.aborted += 1;
        self.events.push(TraceEvent::Aborted { transition: id });
        self.refresh_capacity_metrics();
        Ok(())
    }

    fn release_transition_refs(&mut self, transition: &PreparedTransition) {
        for representation in &transition.representations {
            self.representations
                .get_mut(representation)
                .unwrap()
                .references
                .reservations -= 1;
        }
    }

    pub fn demote_to_host(&mut self, id: PhysicalRepresentationId) -> Result<(), Error> {
        let representation = self
            .representations
            .get(&id)
            .ok_or(Error::RepresentationNotFound)?;
        if representation.references.protected() {
            return Err(Error::Protected);
        }
        let bytes = representation.bytes;
        if !representation.host {
            self.ensure_host_capacity(bytes)?;
        }
        let representation = self.representations.get_mut(&id).unwrap();
        representation.host = true;
        if representation.device {
            representation.device = false;
            self.metrics.demotions += 1;
            self.events.push(TraceEvent::Demoted {
                representation: id,
                bytes,
            });
        }
        self.refresh_capacity_metrics();
        Ok(())
    }

    pub fn evict(&mut self, id: PhysicalRepresentationId, tier: Tier) -> Result<(), Error> {
        let representation = self
            .representations
            .get(&id)
            .ok_or(Error::RepresentationNotFound)?;
        if representation.references.protected() {
            return Err(Error::Protected);
        }
        let bytes = representation.bytes;
        let representation = self.representations.get_mut(&id).unwrap();
        match tier {
            Tier::Device if representation.device => representation.device = false,
            Tier::Host if representation.host => representation.host = false,
            Tier::Storage if representation.storage => representation.storage = false,
            _ => return Ok(()),
        }
        self.metrics.evictions += 1;
        self.events.push(TraceEvent::Evicted {
            representation: id,
            tier,
            bytes,
        });
        self.refresh_capacity_metrics();
        Ok(())
    }

    pub fn metrics(&self) -> Metrics {
        self.metrics
    }

    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    fn validate_composite(
        &self,
        mapping: EvaluatedPrefixId,
        represented_end: usize,
        required: ComponentMask,
        representations: &[PhysicalRepresentationId],
    ) -> Result<(), Error> {
        let mut present = Vec::new();
        for id in representations {
            let representation = self
                .representations
                .get(id)
                .ok_or(Error::RepresentationNotFound)?;
            if representation.mapping != mapping
                || representation.represented_end != represented_end
                || present.contains(&representation.component)
            {
                return Err(Error::InvalidComposite);
            }
            present.push(representation.component);
        }
        for component in [
            Component::GlobalKv,
            Component::SlidingWindow,
            Component::Recurrent,
        ] {
            if required.contains(component.mask()) != present.contains(&component) {
                return Err(Error::InvalidComposite);
            }
        }
        Ok(())
    }

    fn ensure_host_capacity(&self, additional: usize) -> Result<(), Error> {
        if self
            .host_used()
            .checked_add(additional)
            .is_some_and(|used| used <= self.capacity.host_bytes)
        {
            Ok(())
        } else {
            Err(Error::HostCapacity)
        }
    }

    fn ensure_device_capacity(&mut self, additional_guarded: usize) -> Result<(), Error> {
        let protected = self.protected_device_bytes();
        let reservations: usize = self
            .growth
            .values()
            .map(|reservation| reservation.bytes)
            .sum::<usize>()
            + self
                .transitions
                .values()
                .map(|transition| transition.reserved_bytes)
                .sum::<usize>()
            + self.detached_transfer_reserved_bytes();
        let required = protected
            .checked_add(reservations)
            .and_then(|value| value.checked_add(additional_guarded))
            .ok_or(Error::DeviceCapacity)?;
        if required > self.capacity.device_bytes {
            return Err(Error::DeviceCapacity);
        }
        self.ensure_device_space(additional_guarded)
    }

    fn ensure_device_space(&mut self, additional: usize) -> Result<(), Error> {
        while self
            .device_used()
            .checked_add(additional)
            .is_none_or(|used| used > self.capacity.device_bytes)
        {
            let candidate = self
                .representations
                .values()
                .filter(|representation| {
                    representation.device && !representation.references.protected()
                })
                .min_by_key(|representation| {
                    (
                        representation.reuse_value,
                        representation.last_used,
                        representation.id.0,
                    )
                })
                .map(|representation| representation.id)
                .ok_or(Error::DeviceCapacity)?;
            self.evict(candidate, Tier::Device)?;
        }
        Ok(())
    }

    fn device_used(&self) -> usize {
        self.representations
            .values()
            .filter(|representation| representation.device)
            .map(|representation| representation.bytes)
            .sum()
    }

    fn protected_device_bytes(&self) -> usize {
        self.representations
            .values()
            .filter(|representation| representation.device && representation.references.active != 0)
            .map(|representation| representation.bytes)
            .sum()
    }

    fn detached_transfer_reserved_bytes(&self) -> usize {
        self.transfers
            .values()
            .filter(|transfer| transfer.detached && !transfer.completed)
            .map(|transfer| transfer.bytes)
            .sum()
    }

    fn host_used(&self) -> usize {
        self.representations
            .values()
            .filter(|representation| representation.host)
            .map(|representation| representation.bytes)
            .sum()
    }

    fn refresh_capacity_metrics(&mut self) {
        self.metrics.device_active = self.protected_device_bytes();
        self.metrics.device_growth_reserved = self
            .growth
            .values()
            .map(|reservation| reservation.bytes)
            .sum();
        self.metrics.device_transition_reserved = self
            .transitions
            .values()
            .map(|transition| transition.reserved_bytes)
            .sum();
        self.metrics.device_detached_transfer_reserved = self.detached_transfer_reserved_bytes();
        self.metrics.device_warm = self
            .representations
            .values()
            .filter(|representation| {
                representation.device && !representation.references.protected()
            })
            .map(|representation| representation.bytes)
            .sum();
        self.metrics.host_used = self.host_used();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(value: u8) -> EvaluatedPrefixId {
        EvaluatedPrefixId([value; 32])
    }
    fn context(value: u64) -> LogicalContextId {
        LogicalContextId(value)
    }
    fn all() -> ComponentMask {
        ComponentMask::GLOBAL_KV
            .union(ComponentMask::SWA)
            .union(ComponentMask::RECURRENT)
    }

    fn register_composite(
        manager: &mut PhysicalManager,
        map: EvaluatedPrefixId,
        tier: Tier,
        bytes: usize,
    ) -> Vec<PhysicalRepresentationId> {
        [
            Component::GlobalKv,
            Component::SlidingWindow,
            Component::Recurrent,
        ]
        .into_iter()
        .map(|component| {
            manager
                .register(map, component, 32, bytes, tier, 10)
                .unwrap()
        })
        .collect()
    }

    #[test]
    fn guarded_capacity_evicts_warm_state_deterministically() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 12,
            host_bytes: 20,
        });
        let low = manager
            .register(mapping(1), Component::GlobalKv, 1, 4, Tier::Device, 1)
            .unwrap();
        let high = manager
            .register(mapping(2), Component::GlobalKv, 1, 4, Tier::Device, 9)
            .unwrap();
        let reservation = manager.reserve_growth(6).unwrap();
        assert!(!manager.representation(low).unwrap().device);
        assert!(manager.representation(high).unwrap().device);
        assert_eq!(manager.metrics().device_growth_reserved, 6);
        manager.release_growth(reservation).unwrap();
        assert_eq!(
            manager.release_growth(reservation),
            Err(Error::GrowthReservationNotFound)
        );
    }

    #[test]
    fn host_promotion_and_unprotected_demotion_are_accounted() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 20,
        });
        let reps = register_composite(&mut manager, mapping(3), Tier::Host, 2);
        let (transition, class, transfers) = manager
            .prepare_transition(context(1), 4, mapping(3), 32, all(), &reps, false)
            .unwrap();
        assert_eq!(class, TransitionClass::Quiesced);
        for transfer in transfers {
            manager.complete_transfer(transfer, true).unwrap();
        }
        manager.commit_transition(transition, 4).unwrap();
        assert_eq!(manager.metrics().promotions, 3);
        assert_eq!(manager.demote_to_host(reps[0]), Err(Error::Protected));
        let replacement = register_composite(&mut manager, mapping(4), Tier::Device, 1);
        let (transition, _, _) = manager
            .prepare_transition(context(1), 4, mapping(4), 32, all(), &replacement, false)
            .unwrap();
        manager.commit_transition(transition, 4).unwrap();
        manager.demote_to_host(reps[0]).unwrap();
        assert!(!manager.representation(reps[0]).unwrap().device);
    }

    #[test]
    fn reference_only_switch_and_stale_revision_are_transactional() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 0,
        });
        let first = register_composite(&mut manager, mapping(1), Tier::Device, 1);
        let (transition, class, transfers) = manager
            .prepare_transition(context(7), 2, mapping(1), 32, all(), &first, false)
            .unwrap();
        assert_eq!(class, TransitionClass::ReferenceOnly);
        assert!(transfers.is_empty());
        assert_eq!(
            manager.commit_transition(transition, 3),
            Err(Error::StaleRevision)
        );
        assert!(manager.active_binding(context(7)).is_none());
        manager.commit_transition(transition, 2).unwrap();
        assert_eq!(manager.active_binding(context(7)).unwrap().1, mapping(1));
    }

    #[test]
    fn abort_preserves_binding_and_transfer_owner_until_completion() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 20,
        });
        let first = register_composite(&mut manager, mapping(1), Tier::Device, 1);
        let (transition, _, _) = manager
            .prepare_transition(context(2), 1, mapping(1), 32, all(), &first, false)
            .unwrap();
        manager.commit_transition(transition, 1).unwrap();
        let next = register_composite(&mut manager, mapping(2), Tier::Host, 1);
        let (transition, _, transfers) = manager
            .prepare_transition(context(2), 1, mapping(2), 32, all(), &next, false)
            .unwrap();
        manager.abort_transition(transition).unwrap();
        assert_eq!(manager.active_binding(context(2)).unwrap().1, mapping(1));
        assert_eq!(
            manager
                .representation(next[0])
                .unwrap()
                .references
                .transfers,
            1
        );
        for transfer in transfers {
            manager.complete_transfer(transfer, false).unwrap();
        }
        assert_eq!(
            manager
                .representation(next[0])
                .unwrap()
                .references
                .transfers,
            0
        );
        assert_eq!(manager.metrics().device_detached_transfer_reserved, 0);
    }

    #[test]
    fn detached_transfers_keep_device_capacity_reserved() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 20,
        });
        let first = register_composite(&mut manager, mapping(1), Tier::Device, 1);
        let (transition, _, _) = manager
            .prepare_transition(context(2), 1, mapping(1), 32, all(), &first, false)
            .unwrap();
        manager.commit_transition(transition, 1).unwrap();
        let next = register_composite(&mut manager, mapping(2), Tier::Host, 1);
        let (transition, _, transfers) = manager
            .prepare_transition(context(2), 1, mapping(2), 32, all(), &next, false)
            .unwrap();

        manager.abort_transition(transition).unwrap();

        assert_eq!(manager.metrics().device_detached_transfer_reserved, 3);
        assert_eq!(manager.reserve_growth(15), Err(Error::DeviceCapacity));
        for transfer in transfers {
            manager.complete_transfer(transfer, true).unwrap();
        }
        assert_eq!(manager.metrics().device_detached_transfer_reserved, 0);
    }

    #[test]
    fn failed_transfer_cannot_publish_or_replace_source() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 20,
        });
        let first = register_composite(&mut manager, mapping(1), Tier::Device, 1);
        let (transition, _, _) = manager
            .prepare_transition(context(4), 1, mapping(1), 32, all(), &first, false)
            .unwrap();
        manager.commit_transition(transition, 1).unwrap();
        let next = register_composite(&mut manager, mapping(2), Tier::Host, 1);
        let (transition, _, transfers) = manager
            .prepare_transition(context(4), 1, mapping(2), 32, all(), &next, false)
            .unwrap();
        manager.complete_transfer(transfers[0], false).unwrap();
        assert_eq!(
            manager.commit_transition(transition, 1),
            Err(Error::TransferFailed)
        );
        assert_eq!(manager.active_binding(context(4)).unwrap().1, mapping(1));
        manager.abort_transition(transition).unwrap();
    }

    #[test]
    fn composite_requires_exact_components_mapping_and_boundary() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 20,
            host_bytes: 20,
        });
        let reps = register_composite(&mut manager, mapping(1), Tier::Device, 1);
        assert_eq!(
            manager.prepare_transition(context(1), 0, mapping(1), 31, all(), &reps, false),
            Err(Error::InvalidComposite)
        );
        assert_eq!(
            manager.prepare_transition(context(1), 0, mapping(2), 32, all(), &reps, false),
            Err(Error::InvalidComposite)
        );
        assert_eq!(
            manager.prepare_transition(context(1), 0, mapping(1), 32, all(), &reps[..2], false),
            Err(Error::InvalidComposite)
        );
    }

    #[test]
    fn host_capacity_and_observability_are_explicit() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 4,
            host_bytes: 3,
        });
        let rep = manager
            .register(mapping(1), Component::GlobalKv, 1, 3, Tier::Host, 0)
            .unwrap();
        assert_eq!(
            manager.register(mapping(1), Component::Recurrent, 1, 1, Tier::Host, 0),
            Err(Error::HostCapacity)
        );
        assert_eq!(manager.metrics().host_used, 3);
        manager.evict(rep, Tier::Host).unwrap();
        assert_eq!(manager.metrics().host_used, 0);
        assert!(manager.events().iter().any(|event| matches!(
            event,
            TraceEvent::Evicted {
                tier: Tier::Host,
                ..
            }
        )));
    }

    #[test]
    fn recompute_and_explicit_ownership_lifetimes_work() {
        let mut manager = PhysicalManager::new(Capacity {
            device_bytes: 6,
            host_bytes: 0,
        });
        let reps = register_composite(&mut manager, mapping(9), Tier::Storage, 2);
        for rep in &reps {
            manager.evict(*rep, Tier::Storage).unwrap();
        }
        assert_eq!(
            manager.prepare_transition(context(9), 1, mapping(9), 32, all(), &reps, false),
            Err(Error::NoPromotionSource)
        );
        let (transition, class, transfers) = manager
            .prepare_transition(context(9), 1, mapping(9), 32, all(), &reps, true)
            .unwrap();
        assert_eq!(class, TransitionClass::Recompute);
        assert_eq!(manager.transition_class(transition), Some(class));
        assert!(transfers.is_empty());
        assert_eq!(
            manager.commit_transition(transition, 1),
            Err(Error::TransferPending)
        );
        for rep in &reps {
            manager.complete_recompute(transition, *rep).unwrap();
        }
        manager.commit_transition(transition, 1).unwrap();
        assert_eq!(manager.metrics().recomputed, 3);
        manager.unbind(context(9)).unwrap();
        assert_eq!(manager.unbind(context(9)), Err(Error::BindingNotFound));
        for rep in reps {
            manager.release_logical_reference(rep).unwrap();
            assert!(manager.representation(rep).is_none());
        }
    }
}
