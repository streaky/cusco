use crate::{
    EngineOutput, EngineRequest, Error, ExecutionSession, InferenceEngine, PrioritySource,
    QuantumKind, QuantumObservation, SchedulerPolicyConfig, SchedulingClass, SessionStep,
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender, TrySendError},
    },
    thread,
    time::{Duration, Instant},
};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SchedulerMetrics {
    pub admitted: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub deadlines: u64,
    pub failed: u64,
    pub rounds: u64,
    pub runnable: usize,
    pub waiting_for_consumer: usize,
    pub diagnostic_records: u64,
    pub diagnostic_records_lost: u64,
    pub diagnostic_records_delivered: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchedulerStatus {
    pub policy: SchedulerPolicyConfig,
    pub metrics: SchedulerMetrics,
    pub diagnostic_loss_by_kind: HashMap<String, u64>,
}

#[derive(Clone, Debug, Serialize)]
struct SchedulerDecision {
    #[serde(rename = "type")]
    record_type: &'static str,
    transport_correlation_id: String,
    inference_operation_id: String,
    execution_session_id: String,
    principal: String,
    class: SchedulingClass,
    priority_source: PrioritySource,
    model_id: String,
    model_epoch: u64,
    queue_age_ms: u128,
    queue_age_rounds: u64,
    age_promotions: u8,
    quantum_kind: QuantumKind,
    charged_tokens: usize,
    quantum_duration_ns: u128,
    context_placement: String,
    executor_slot_occupied: bool,
    transition_cost_bytes: u64,
    capacity_reserved_bytes: u64,
    cancelled: bool,
    deadline_expired: bool,
    reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct DiagnosticLossSummary {
    #[serde(rename = "type")]
    record_type: &'static str,
    total: u64,
    by_kind: HashMap<String, u64>,
}

#[derive(Default)]
struct DiagnosticLoss {
    pending_total: u64,
    pending_by_kind: HashMap<String, u64>,
    all_by_kind: HashMap<String, u64>,
}

struct SchedulerDiagnostics {
    sender: SyncSender<String>,
    emitted: AtomicU64,
    delivered: Arc<AtomicU64>,
    lost: AtomicU64,
    loss: Mutex<DiagnosticLoss>,
}

impl SchedulerDiagnostics {
    fn stderr(capacity: usize) -> Arc<Self> {
        Self::with_sink(capacity, |line| {
            let mut stderr = std::io::stderr().lock();
            let _ = writeln!(stderr, "{line}");
        })
    }

    fn with_sink(capacity: usize, sink: impl Fn(&str) + Send + 'static) -> Arc<Self> {
        let delivered = Arc::new(AtomicU64::new(0));
        let worker_delivered = delivered.clone();
        let (sender, receiver) = mpsc::sync_channel::<String>(capacity);
        thread::Builder::new()
            .name("cusco-scheduler-diagnostics".into())
            .spawn(move || {
                while let Ok(line) = receiver.recv() {
                    sink(&line);
                    worker_delivered.fetch_add(1, Ordering::Release);
                }
            })
            .expect("scheduler diagnostic worker starts");
        Arc::new(Self {
            sender,
            emitted: AtomicU64::new(0),
            lost: AtomicU64::new(0),
            delivered,
            loss: Mutex::new(DiagnosticLoss::default()),
        })
    }

    fn emit<T: Serialize>(&self, kind: &str, record: &T) {
        let Ok(line) = serde_json::to_string(record) else {
            self.record_loss("serialization_error");
            return;
        };
        self.flush_loss_summary();
        match self.sender.try_send(line) {
            Ok(()) => {
                self.emitted.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.record_loss(kind);
            }
        }
    }

    fn flush_loss_summary(&self) {
        let summary = {
            let loss = self.loss.lock();
            if loss.pending_total == 0 {
                return;
            }
            DiagnosticLossSummary {
                record_type: "scheduler_diagnostic_loss",
                total: loss.pending_total,
                by_kind: loss.pending_by_kind.clone(),
            }
        };
        let Ok(line) = serde_json::to_string(&summary) else {
            return;
        };
        if self.sender.try_send(line).is_ok() {
            self.emitted.fetch_add(1, Ordering::Relaxed);
            let mut loss = self.loss.lock();
            loss.pending_total = 0;
            loss.pending_by_kind.clear();
        }
    }

    fn record_loss(&self, kind: &str) {
        self.lost.fetch_add(1, Ordering::Relaxed);
        let mut loss = self.loss.lock();
        loss.pending_total = loss.pending_total.saturating_add(1);
        *loss.pending_by_kind.entry(kind.into()).or_default() += 1;
        *loss.all_by_kind.entry(kind.into()).or_default() += 1;
    }

    fn loss_by_kind(&self) -> HashMap<String, u64> {
        self.loss.lock().all_by_kind.clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FlowKey {
    principal: String,
    class: SchedulingClass,
}

struct PolicyEntry {
    id: u64,
    flow: FlowKey,
    admitted_round: u64,
    admitted_at: Instant,
}

struct Selection {
    id: u64,
    queue_age_rounds: u64,
    queue_age_ms: u128,
    promotions: u8,
    charge_scale: i64,
}

struct FairPolicy {
    config: SchedulerPolicyConfig,
    runnable: VecDeque<PolicyEntry>,
    deficits: HashMap<FlowKey, i64>,
    round: u64,
    cursor: usize,
}

impl FairPolicy {
    fn new(config: SchedulerPolicyConfig) -> Self {
        Self {
            config,
            runnable: VecDeque::new(),
            deficits: HashMap::new(),
            round: 0,
            cursor: 0,
        }
    }

    fn enqueue(&mut self, id: u64, flow: FlowKey, admitted_round: u64, admitted_at: Instant) {
        self.runnable.push_back(PolicyEntry {
            id,
            flow,
            admitted_round,
            admitted_at,
        });
    }

    fn effective_class(&self, entry: &PolicyEntry) -> (SchedulingClass, u8) {
        let promotions = ((self.round.saturating_sub(entry.admitted_round))
            / self.config.promotion_rounds)
            .min(2) as u8;
        let class = match (entry.flow.class, promotions) {
            (SchedulingClass::Batch, 1) => SchedulingClass::Standard,
            (SchedulingClass::Batch, 2) | (SchedulingClass::Standard, 1..) => {
                SchedulingClass::Interactive
            }
            (class, _) => class,
        };
        (class, promotions)
    }

    fn weight(&self, class: SchedulingClass) -> i64 {
        let weight = match class {
            SchedulingClass::Interactive => self.config.interactive_weight,
            SchedulingClass::Standard => self.config.standard_weight,
            SchedulingClass::Batch => self.config.batch_weight,
        };
        i64::from(weight.saturating_mul(self.config.deficit_refill))
    }

    fn select<F>(&mut self, mut available: F) -> Option<Selection>
    where
        F: FnMut(u64) -> bool,
    {
        if self.runnable.is_empty() {
            return None;
        }
        let mut heads = Vec::<(usize, FlowKey, SchedulingClass, u8, i64)>::new();
        for (index, entry) in self.runnable.iter().enumerate() {
            if heads.iter().any(|(_, flow, _, _, _)| flow == &entry.flow) {
                continue;
            }
            if !available(entry.id) {
                continue;
            }
            let (class, promotions) = self.effective_class(entry);
            let credit = self.weight(class);
            heads.push((index, entry.flow.clone(), class, promotions, credit));
        }
        if heads.is_empty() {
            return None;
        }
        self.round = self.round.saturating_add(1);
        for (_, flow, _, _, credit) in &heads {
            *self.deficits.entry(flow.clone()).or_default() = self
                .deficits
                .get(flow)
                .copied()
                .unwrap_or_default()
                .saturating_add(*credit)
                .min(credit.saturating_mul(8));
        }
        let total_weight = heads
            .iter()
            .map(|(_, _, _, _, weight)| *weight)
            .sum::<i64>()
            .max(1);
        let max_deficit = heads
            .iter()
            .map(|(_, flow, _, _, _)| self.deficits.get(flow).copied().unwrap_or_default())
            .max()
            .unwrap_or_default();
        let eligible = heads
            .iter()
            .filter(|(_, flow, _, _, _)| {
                self.deficits.get(flow).copied().unwrap_or_default() == max_deficit
            })
            .collect::<Vec<_>>();
        let selected = eligible[self.cursor % eligible.len()];
        self.cursor = (self.cursor + 1) % eligible.len();
        let entry = self
            .runnable
            .remove(selected.0)
            .expect("selected entry exists");
        Some(Selection {
            id: entry.id,
            queue_age_rounds: self.round.saturating_sub(entry.admitted_round),
            queue_age_ms: entry.admitted_at.elapsed().as_millis(),
            promotions: selected.3,
            charge_scale: total_weight,
        })
    }

    fn charge(&mut self, flow: &FlowKey, tokens: usize, scale: i64) {
        *self.deficits.entry(flow.clone()).or_default() -=
            (tokens.max(1) as i64).saturating_mul(scale);
    }

    fn clear_if_idle(&mut self, flow: &FlowKey, jobs: &HashMap<u64, Job>) {
        if !self.runnable.iter().any(|entry| &entry.flow == flow)
            && !jobs.values().any(|job| &job.flow == flow)
        {
            self.deficits.remove(flow);
        }
    }

    fn remove(&mut self, id: u64) -> bool {
        let previous = self.runnable.len();
        self.runnable.retain(|entry| entry.id != id);
        self.runnable.len() != previous
    }
}

enum ClientMessage {
    Token {
        id: i32,
        piece: Vec<u8>,
        terminal_or_control: bool,
    },
    Finished(EngineOutput),
    Failed(Error),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ModelSlotKey {
    model_id: String,
    model_epoch: u64,
}

enum Command {
    Submit {
        id: u64,
        request: Box<EngineRequest>,
        response: Sender<ClientMessage>,
    },
    Continue(u64),
    Stop(u64),
    Cancel(u64),
    Shutdown,
}

enum SchedulerEvent {
    Client(Command),
    SlotCompleted(Box<SlotCompletion>),
}

enum SlotAction {
    Start(Box<EngineRequest>),
    Step(Box<dyn ExecutionSession>),
    Finish(Box<dyn ExecutionSession>),
    Shutdown,
}

struct SlotTask {
    id: u64,
    action: SlotAction,
    selection: Option<Selection>,
}
struct SlotCompletion {
    id: u64,
    result: Result<SessionStep, Error>,
    session: Option<Box<dyn ExecutionSession>>,
    selection: Option<Selection>,
    quantum_duration_ns: u128,
}
struct SlotWorker {
    sender: Sender<SlotTask>,
    handle: thread::JoinHandle<()>,
    busy: bool,
}

struct Job {
    request: Option<EngineRequest>,
    session: Option<Box<dyn ExecutionSession>>,
    control: Arc<crate::RequestControl>,
    response: Sender<ClientMessage>,
    flow: FlowKey,
    scheduling: crate::SchedulingMetadata,
    slot: ModelSlotKey,
    execution_session_id: String,
    admitted_round: u64,
    admitted_at: Instant,
    in_flight: bool,
    cancel_requested: bool,
}

struct SchedulerCounters {
    admitted: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    deadlines: AtomicU64,
    failed: AtomicU64,
    rounds: AtomicU64,
    runnable: AtomicUsize,
    waiting: AtomicUsize,
}

impl Default for SchedulerCounters {
    fn default() -> Self {
        Self {
            admitted: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            deadlines: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            rounds: AtomicU64::new(0),
            runnable: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
        }
    }
}

pub struct WorkloadScheduler {
    inner: Arc<dyn InferenceEngine>,
    commands: Sender<SchedulerEvent>,
    next_id: AtomicU64,
    policy: SchedulerPolicyConfig,
    counters: Arc<SchedulerCounters>,
    diagnostics: Arc<SchedulerDiagnostics>,
    worker: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WorkloadScheduler {
    pub fn new(
        inner: Arc<dyn InferenceEngine>,
        policy: SchedulerPolicyConfig,
    ) -> Result<Arc<Self>, Error> {
        Self::new_with_diagnostics(inner, policy, None::<fn(&str)>)
    }
    pub fn new_with_diagnostics<F>(
        inner: Arc<dyn InferenceEngine>,
        policy: SchedulerPolicyConfig,
        sink: Option<F>,
    ) -> Result<Arc<Self>, Error>
    where
        F: Fn(&str) + Send + 'static,
    {
        let policy = policy.validate()?;
        let diagnostics = match sink {
            Some(sink) => SchedulerDiagnostics::with_sink(policy.diagnostic_capacity, sink),
            None => SchedulerDiagnostics::stderr(policy.diagnostic_capacity),
        };
        let counters = Arc::new(SchedulerCounters::default());
        let (events, receiver) = mpsc::channel();
        let worker_inner = inner.clone();
        let worker_counters = counters.clone();
        let worker_diagnostics = diagnostics.clone();
        let worker_events = events.clone();
        let worker = thread::Builder::new()
            .name("cusco-workload-scheduler".into())
            .spawn(move || {
                run_scheduler(
                    worker_inner,
                    policy,
                    worker_events,
                    receiver,
                    worker_counters,
                    worker_diagnostics,
                )
            })
            .map_err(|error| Error::State(error.to_string()))?;
        Ok(Arc::new(Self {
            inner,
            commands: events,
            next_id: AtomicU64::new(0),
            policy,
            counters,
            diagnostics,
            worker: Mutex::new(Some(worker)),
        }))
    }

    pub fn status(&self) -> SchedulerStatus {
        SchedulerStatus {
            policy: self.policy,
            metrics: SchedulerMetrics {
                admitted: self.counters.admitted.load(Ordering::Acquire),
                completed: self.counters.completed.load(Ordering::Acquire),
                cancelled: self.counters.cancelled.load(Ordering::Acquire),
                deadlines: self.counters.deadlines.load(Ordering::Acquire),
                failed: self.counters.failed.load(Ordering::Acquire),
                rounds: self.counters.rounds.load(Ordering::Acquire),
                runnable: self.counters.runnable.load(Ordering::Acquire),
                waiting_for_consumer: self.counters.waiting.load(Ordering::Acquire),
                diagnostic_records: self.diagnostics.emitted.load(Ordering::Acquire),
                diagnostic_records_lost: self.diagnostics.lost.load(Ordering::Acquire),
                diagnostic_records_delivered: self.diagnostics.delivered.load(Ordering::Acquire),
            },
            diagnostic_loss_by_kind: self.diagnostics.loss_by_kind(),
        }
    }

    pub fn flush_diagnostics(&self, timeout: Duration) -> bool {
        let target = self.diagnostics.emitted.load(Ordering::Acquire);
        let started = Instant::now();
        while self.diagnostics.delivered.load(Ordering::Acquire) < target {
            if started.elapsed() >= timeout {
                return false;
            }
            thread::sleep(Duration::from_millis(1));
        }
        true
    }
}

impl Drop for WorkloadScheduler {
    fn drop(&mut self) {
        let _ = self
            .commands
            .send(SchedulerEvent::Client(Command::Shutdown));
        if let Some(worker) = self.worker.lock().take() {
            let _ = worker.join();
        }
    }
}

struct SchedulerClientSession {
    id: u64,
    control: Arc<crate::RequestControl>,
    commands: Sender<SchedulerEvent>,
    receiver: Receiver<ClientMessage>,
    waiting_ack: bool,
    finished: Option<EngineOutput>,
}

impl SchedulerClientSession {
    fn receive(&mut self) -> Result<SessionStep, Error> {
        match self.receiver.recv().map_err(|_| Error::Cancelled)? {
            ClientMessage::Token {
                id,
                piece,
                terminal_or_control,
            } => {
                self.waiting_ack = true;
                Ok(SessionStep::Token {
                    id,
                    piece,
                    terminal_or_control,
                    observation: QuantumObservation::model_free(QuantumKind::Decode, 1),
                })
            }
            ClientMessage::Finished(output) => {
                self.finished = Some(output.clone());
                Ok(SessionStep::Finished(output))
            }
            ClientMessage::Failed(error) => Err(error),
        }
    }
}

impl ExecutionSession for SchedulerClientSession {
    fn step(&mut self) -> Result<SessionStep, Error> {
        if let Some(output) = &self.finished {
            return Ok(SessionStep::Finished(output.clone()));
        }
        if self.waiting_ack {
            self.commands
                .send(SchedulerEvent::Client(Command::Continue(self.id)))
                .map_err(|_| Error::Cancelled)?;
            self.waiting_ack = false;
        }
        self.receive()
    }

    fn finish(&mut self) -> Result<EngineOutput, Error> {
        if let Some(output) = &self.finished {
            return Ok(output.clone());
        }
        if self.waiting_ack {
            self.commands
                .send(SchedulerEvent::Client(Command::Stop(self.id)))
                .map_err(|_| Error::Cancelled)?;
            self.waiting_ack = false;
        } else {
            self.commands
                .send(SchedulerEvent::Client(Command::Cancel(self.id)))
                .map_err(|_| Error::Cancelled)?;
        }
        match self.receive()? {
            SessionStep::Finished(output) => Ok(output),
            _ => Err(Error::State("scheduler returned work after stop".into())),
        }
    }
}

impl Drop for SchedulerClientSession {
    fn drop(&mut self) {
        if self.finished.is_none() {
            self.control.cancel();
            let _ = self
                .commands
                .send(SchedulerEvent::Client(Command::Cancel(self.id)));
        }
    }
}

impl InferenceEngine for WorkloadScheduler {
    fn start_session(
        &self,
        mut request: EngineRequest,
    ) -> Result<Box<dyn ExecutionSession>, Error> {
        request.control.check()?;
        if request.scheduling.correlation_id.is_empty() {
            request.scheduling.correlation_id = Uuid::new_v4().to_string();
        }
        if request.scheduling.inference_id.is_empty() {
            request.scheduling.inference_id = Uuid::new_v4().to_string();
        }
        request.prefill_chunk_tokens = self.policy.prefill_tokens;
        let control = request.control.clone();
        let id = self
            .next_id
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
            .map_err(|_| Error::State("scheduler session id space exhausted".into()))?;
        let (response, receiver) = mpsc::channel();
        self.commands
            .send(SchedulerEvent::Client(Command::Submit {
                id,
                request: Box::new(request),
                response,
            }))
            .map_err(|_| Error::ShuttingDown)?;
        Ok(Box::new(SchedulerClientSession {
            id,
            control,
            commands: self.commands.clone(),
            receiver,
            waiting_ack: false,
            finished: None,
        }))
    }
    fn prepare_model(&self, model: &crate::ModelRecord) -> Result<(), Error> {
        self.inner.prepare_model(model)
    }

    fn commit_model(&self, model: &crate::ModelRecord, replaced_epoch: Option<u64>) {
        self.inner.commit_model(model, replaced_epoch);
    }

    fn retire_model(&self, id: &str, epoch: u64) {
        self.inner.retire_model(id, epoch);
    }

    fn demote_inactive(&self) -> Result<usize, Error> {
        self.inner.demote_inactive()
    }

    fn residency_status(&self) -> Option<Value> {
        let mut status = self.inner.residency_status().unwrap_or_else(|| json!({}));
        if let Some(object) = status.as_object_mut() {
            object.insert(
                "scheduler".into(),
                serde_json::to_value(self.status()).expect("scheduler status serializes"),
            );
        }
        Some(status)
    }
}

fn run_scheduler(
    inner: Arc<dyn InferenceEngine>,
    config: SchedulerPolicyConfig,
    events: Sender<SchedulerEvent>,
    receiver: Receiver<SchedulerEvent>,
    counters: Arc<SchedulerCounters>,
    diagnostics: Arc<SchedulerDiagnostics>,
) {
    let mut policy = FairPolicy::new(config);
    let mut jobs = HashMap::<u64, Job>::new();
    let mut slots = HashMap::<ModelSlotKey, SlotWorker>::new();
    let mut shutting_down = false;
    loop {
        while dispatch_one(
            &inner,
            &events,
            &mut slots,
            &mut jobs,
            &mut policy,
            &counters,
        ) {}
        counters
            .runnable
            .store(policy.runnable.len(), Ordering::Release);
        if shutting_down && slots.values().all(|slot| !slot.busy) {
            cancel_all_jobs(&mut jobs);
            shutdown_slots(&mut slots);
            return;
        }
        let Ok(event) = receiver.recv() else {
            cancel_all_jobs(&mut jobs);
            shutdown_slots(&mut slots);
            return;
        };
        match event {
            SchedulerEvent::Client(Command::Shutdown) => {
                shutting_down = true;
                policy.runnable.clear();
                for job in jobs.values_mut() {
                    job.control.cancel();
                    job.cancel_requested = true;
                }
            }
            SchedulerEvent::Client(command) if !shutting_down => handle_command(
                command,
                &inner,
                &events,
                &mut slots,
                &mut jobs,
                &mut policy,
                &counters,
            ),
            SchedulerEvent::Client(Command::Submit { response, .. }) => {
                let _ = response.send(ClientMessage::Failed(Error::ShuttingDown));
            }
            SchedulerEvent::Client(_) => {}
            SchedulerEvent::SlotCompleted(completion) => handle_completion(
                *completion,
                &mut slots,
                &mut jobs,
                &mut policy,
                &counters,
                &diagnostics,
            ),
        }
    }
}

fn spawn_slot(
    key: &ModelSlotKey,
    inner: Arc<dyn InferenceEngine>,
    events: Sender<SchedulerEvent>,
) -> Result<SlotWorker, Error> {
    let (sender, receiver) = mpsc::channel::<SlotTask>();
    let name = format!("cusco-model-slot-{}-{}", key.model_id, key.model_epoch);
    let handle = thread::Builder::new()
        .name(name)
        .spawn(move || {
            while let Ok(task) = receiver.recv() {
                if matches!(task.action, SlotAction::Shutdown) {
                    return;
                }
                let started = Instant::now();
                let (result, session) = match task.action {
                    SlotAction::Start(request) => match inner.start_session(*request) {
                        Ok(session) => (
                            Ok(SessionStep::Progress(QuantumObservation::model_free(
                                QuantumKind::Preparation,
                                1,
                            ))),
                            Some(session),
                        ),
                        Err(error) => (Err(error), None),
                    },
                    SlotAction::Step(mut session) => {
                        let result = session.step();
                        (result, Some(session))
                    }
                    SlotAction::Finish(mut session) => {
                        let result = session.finish().map(SessionStep::Finished);
                        (result, None)
                    }
                    SlotAction::Shutdown => unreachable!(),
                };
                if events
                    .send(SchedulerEvent::SlotCompleted(Box::new(SlotCompletion {
                        id: task.id,
                        result,
                        session,
                        selection: task.selection,
                        quantum_duration_ns: started.elapsed().as_nanos(),
                    })))
                    .is_err()
                {
                    return;
                }
            }
        })
        .map_err(|error| Error::State(error.to_string()))?;
    Ok(SlotWorker {
        sender,
        handle,
        busy: false,
    })
}

fn ensure_slot<'a>(
    key: &ModelSlotKey,
    inner: &Arc<dyn InferenceEngine>,
    events: &Sender<SchedulerEvent>,
    slots: &'a mut HashMap<ModelSlotKey, SlotWorker>,
) -> Result<&'a mut SlotWorker, Error> {
    if !slots.contains_key(key) {
        let worker = spawn_slot(key, inner.clone(), events.clone())?;
        slots.insert(key.clone(), worker);
    }
    Ok(slots.get_mut(key).expect("slot was inserted"))
}

fn dispatch_one(
    inner: &Arc<dyn InferenceEngine>,
    events: &Sender<SchedulerEvent>,
    slots: &mut HashMap<ModelSlotKey, SlotWorker>,
    jobs: &mut HashMap<u64, Job>,
    policy: &mut FairPolicy,
    counters: &SchedulerCounters,
) -> bool {
    let selection = policy.select(|id| {
        jobs.get(&id).is_some_and(|job| {
            !job.in_flight && !slots.get(&job.slot).is_some_and(|slot| slot.busy)
        })
    });
    let Some(selection) = selection else {
        return false;
    };
    let id = selection.id;
    let job = jobs
        .get_mut(&id)
        .expect("policy only selects retained jobs");
    let action = if let Some(request) = job.request.take() {
        SlotAction::Start(Box::new(request))
    } else {
        SlotAction::Step(job.session.take().expect("started job retains session"))
    };
    job.in_flight = true;
    match ensure_slot(&job.slot, inner, events, slots).and_then(|slot| {
        slot.busy = true;
        slot.sender
            .send(SlotTask {
                id,
                action,
                selection: Some(selection),
            })
            .map_err(|_| Error::State("model slot worker stopped".into()))
    }) {
        Ok(()) => true,
        Err(error) => {
            let job = jobs.remove(&id).expect("failed job exists");
            send_terminal(job, Err(error), counters);
            true
        }
    }
}

fn handle_command(
    command: Command,
    inner: &Arc<dyn InferenceEngine>,
    events: &Sender<SchedulerEvent>,
    slots: &mut HashMap<ModelSlotKey, SlotWorker>,
    jobs: &mut HashMap<u64, Job>,
    policy: &mut FairPolicy,
    counters: &SchedulerCounters,
) {
    match command {
        Command::Submit {
            id,
            request,
            response,
        } => {
            let request = *request;
            let flow = FlowKey {
                principal: request.scheduling.principal.clone(),
                class: request.scheduling.class,
            };
            let admitted_at = Instant::now();
            let admitted_round = policy.round;
            let scheduling = request.scheduling.clone();
            let slot = ModelSlotKey {
                model_id: request.model.id.clone(),
                model_epoch: request.model.epoch,
            };
            let control = request.control.clone();
            jobs.insert(
                id,
                Job {
                    request: Some(request),
                    session: None,
                    control,
                    response,
                    flow: flow.clone(),
                    scheduling,
                    slot,
                    execution_session_id: Uuid::new_v4().to_string(),
                    admitted_round,
                    admitted_at,
                    in_flight: false,
                    cancel_requested: false,
                },
            );
            counters.admitted.fetch_add(1, Ordering::Relaxed);
            policy.enqueue(id, flow, admitted_round, admitted_at);
        }
        Command::Continue(id) => {
            if let Some(job) = jobs.get(&id) {
                policy.enqueue(id, job.flow.clone(), job.admitted_round, job.admitted_at);
                counters.waiting.fetch_sub(1, Ordering::AcqRel);
            }
        }
        Command::Stop(id) => {
            counters.waiting.fetch_sub(1, Ordering::AcqRel);
            policy.remove(id);
            let Some(job) = jobs.get_mut(&id) else { return };
            let Some(session) = job.session.take() else {
                job.control.cancel();
                job.cancel_requested = true;
                return;
            };
            job.in_flight = true;
            match ensure_slot(&job.slot, inner, events, slots) {
                Ok(slot) => {
                    slot.busy = true;
                    if slot
                        .sender
                        .send(SlotTask {
                            id,
                            action: SlotAction::Finish(session),
                            selection: None,
                        })
                        .is_err()
                    {
                        let job = jobs.remove(&id).expect("stopped job exists");
                        send_terminal(
                            job,
                            Err(Error::State("model slot worker stopped".into())),
                            counters,
                        );
                    }
                }
                Err(error) => {
                    let job = jobs.remove(&id).expect("stopped job exists");
                    send_terminal(job, Err(error), counters);
                }
            }
        }
        Command::Cancel(id) => {
            let was_runnable = policy.remove(id);
            let Some(job) = jobs.get_mut(&id) else { return };
            job.control.cancel();
            if job.in_flight {
                job.cancel_requested = true;
            } else {
                let job = jobs.remove(&id).expect("cancelled job exists");
                if job.session.is_some() && !was_runnable {
                    counters.waiting.fetch_sub(1, Ordering::AcqRel);
                }
                let flow = job.flow.clone();
                send_terminal(job, Err(Error::Cancelled), counters);
                policy.clear_if_idle(&flow, jobs);
            }
        }
        Command::Shutdown => unreachable!(),
    }
}

fn handle_completion(
    completion: SlotCompletion,
    slots: &mut HashMap<ModelSlotKey, SlotWorker>,
    jobs: &mut HashMap<u64, Job>,
    policy: &mut FairPolicy,
    counters: &SchedulerCounters,
    diagnostics: &SchedulerDiagnostics,
) {
    let Some(job) = jobs.get_mut(&completion.id) else {
        return;
    };
    if let Some(slot) = slots.get_mut(&job.slot) {
        slot.busy = false;
    }
    job.in_flight = false;
    job.session = completion.session;
    let mut terminal = None;
    let mut wait_for_consumer = false;
    let observation = match completion.result {
        Ok(SessionStep::Progress(observation)) => observation,
        Ok(SessionStep::Token {
            id,
            piece,
            terminal_or_control,
            observation,
        }) => {
            if job.cancel_requested
                || job
                    .response
                    .send(ClientMessage::Token {
                        id,
                        piece,
                        terminal_or_control,
                    })
                    .is_err()
            {
                terminal = Some(Err(Error::Cancelled));
            } else {
                counters.waiting.fetch_add(1, Ordering::AcqRel);
                wait_for_consumer = true;
            }
            observation
        }
        Ok(SessionStep::Finished(output)) => {
            terminal = Some(if job.cancel_requested {
                Err(Error::Cancelled)
            } else {
                Ok(output)
            });
            QuantumObservation::model_free(QuantumKind::Publication, 1)
        }
        Err(error) => {
            terminal = Some(if job.cancel_requested {
                Err(Error::Cancelled)
            } else {
                Err(error)
            });
            QuantumObservation::model_free(QuantumKind::Preparation, 1)
        }
    };
    if job.cancel_requested && terminal.is_none() {
        terminal = Some(Err(Error::Cancelled));
    }
    if let Some(selection) = completion.selection {
        policy.charge(
            &job.flow,
            observation.charged_tokens,
            selection.charge_scale,
        );
        diagnostics.emit(
            "scheduler_decision",
            &SchedulerDecision {
                record_type: "scheduler_decision",
                transport_correlation_id: job.scheduling.correlation_id.clone(),
                inference_operation_id: job.scheduling.inference_id.clone(),
                execution_session_id: job.execution_session_id.clone(),
                principal: job.scheduling.principal.clone(),
                class: job.scheduling.class,
                priority_source: job.scheduling.source,
                model_id: job.slot.model_id.clone(),
                model_epoch: job.slot.model_epoch,
                queue_age_ms: selection.queue_age_ms,
                queue_age_rounds: selection.queue_age_rounds,
                age_promotions: selection.promotions,
                quantum_kind: observation.kind,
                charged_tokens: observation.charged_tokens,
                quantum_duration_ns: completion.quantum_duration_ns,
                context_placement: observation.context_placement,
                executor_slot_occupied: observation.executor_slot_occupied,
                transition_cost_bytes: observation.transition_cost_bytes,
                capacity_reserved_bytes: observation.capacity_reserved_bytes,
                cancelled: matches!(terminal, Some(Err(Error::Cancelled))),
                deadline_expired: matches!(terminal, Some(Err(Error::Deadline))),
                reason: if selection.promotions > 0 {
                    "age_promoted"
                } else {
                    "class_ready"
                },
            },
        );
    }
    if let Some(result) = terminal {
        let job = jobs.remove(&completion.id).expect("terminal job exists");
        let flow = job.flow.clone();
        let slot = job.slot.clone();
        send_terminal(job, result, counters);
        policy.clear_if_idle(&flow, jobs);
        retire_slot_if_idle(&slot, slots, jobs);
    } else if !wait_for_consumer {
        policy.enqueue(
            completion.id,
            job.flow.clone(),
            job.admitted_round,
            job.admitted_at,
        );
    }
}

fn retire_slot_if_idle(
    key: &ModelSlotKey,
    slots: &mut HashMap<ModelSlotKey, SlotWorker>,
    jobs: &HashMap<u64, Job>,
) {
    if jobs.values().any(|job| &job.slot == key) {
        return;
    }
    if let Some(slot) = slots.remove(key) {
        let _ = slot.sender.send(SlotTask {
            id: 0,
            action: SlotAction::Shutdown,
            selection: None,
        });
        let _ = slot.handle.join();
    }
}

fn cancel_all_jobs(jobs: &mut HashMap<u64, Job>) {
    for (_, job) in jobs.drain() {
        job.control.cancel();
        let _ = job.response.send(ClientMessage::Failed(Error::Cancelled));
    }
}

fn shutdown_slots(slots: &mut HashMap<ModelSlotKey, SlotWorker>) {
    for (_, slot) in slots.drain() {
        let _ = slot.sender.send(SlotTask {
            id: 0,
            action: SlotAction::Shutdown,
            selection: None,
        });
        let _ = slot.handle.join();
    }
}

fn send_terminal(job: Job, result: Result<EngineOutput, Error>, counters: &SchedulerCounters) {
    match result {
        Ok(output) => {
            let _ = job.response.send(ClientMessage::Finished(output));
            counters.completed.fetch_add(1, Ordering::Relaxed);
        }
        Err(Error::Cancelled) => {
            let _ = job.response.send(ClientMessage::Failed(Error::Cancelled));
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        Err(Error::Deadline) => {
            let _ = job.response.send(ClientMessage::Failed(Error::Deadline));
            counters.deadlines.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let _ = job.response.send(ClientMessage::Failed(error));
            counters.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeterministicEngine, ModelRecord, RequestControl};

    #[derive(Default)]
    struct BlockingEngine {
        active: AtomicUsize,
        max_active: AtomicUsize,
        active_by_model: Mutex<HashMap<String, usize>>,
        max_by_model: Mutex<HashMap<String, usize>>,
    }

    impl InferenceEngine for BlockingEngine {
        fn start_session(
            &self,
            request: EngineRequest,
        ) -> Result<Box<dyn ExecutionSession>, Error> {
            let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
            self.max_active.fetch_max(active, Ordering::AcqRel);
            {
                let mut by_model = self.active_by_model.lock();
                let current = by_model.entry(request.model.id.clone()).or_default();
                *current += 1;
                let mut maxima = self.max_by_model.lock();
                maxima
                    .entry(request.model.id.clone())
                    .and_modify(|maximum| *maximum = (*maximum).max(*current))
                    .or_insert(*current);
            }
            thread::sleep(Duration::from_millis(50));
            self.active.fetch_sub(1, Ordering::AcqRel);
            *self
                .active_by_model
                .lock()
                .get_mut(&request.model.id)
                .expect("model activity exists") -= 1;
            Ok(Box::new(ImmediateSession {
                output: EngineOutput::default(),
            }))
        }
    }

    struct ImmediateSession {
        output: EngineOutput,
    }

    impl ExecutionSession for ImmediateSession {
        fn step(&mut self) -> Result<SessionStep, Error> {
            Ok(SessionStep::Finished(self.output.clone()))
        }

        fn finish(&mut self) -> Result<EngineOutput, Error> {
            Ok(self.output.clone())
        }
    }
    use parking_lot::Mutex;
    use std::{path::PathBuf, time::Duration};

    fn request(class: SchedulingClass, principal: &str, words: usize) -> EngineRequest {
        EngineRequest {
            model: ModelRecord {
                id: "m".into(),
                revision: "r".into(),
                path: PathBuf::from("m.gguf"),
                sha256: String::new(),
                aliases: vec![],
                family: "gemma4".into(),
                size_bytes: 1,
                epoch: 1,
            },
            prompt: std::iter::repeat_n("x", words)
                .collect::<Vec<_>>()
                .join(" "),
            max_tokens: 2,
            prior_tokens: vec![],
            sampling: Default::default(),
            control: Arc::new(RequestControl::new()),
            scheduling: crate::SchedulingMetadata {
                class,
                source: PrioritySource::ControlledWorkload,
                principal: principal.into(),
                correlation_id: Uuid::new_v4().to_string(),
                inference_id: Uuid::new_v4().to_string(),
            },
            prefill_chunk_tokens: 1,
        }
    }

    #[test]
    fn policy_preserves_flow_fifo_and_promotes_waiting_batch_work() {
        let config = SchedulerPolicyConfig {
            promotion_rounds: 2,
            ..Default::default()
        };
        let mut policy = FairPolicy::new(config);
        let now = Instant::now();
        policy.enqueue(
            1,
            FlowKey {
                principal: "p".into(),
                class: SchedulingClass::Batch,
            },
            0,
            now,
        );
        policy.enqueue(
            2,
            FlowKey {
                principal: "p".into(),
                class: SchedulingClass::Batch,
            },
            0,
            now,
        );
        for id in 10..14 {
            policy.enqueue(
                id,
                FlowKey {
                    principal: format!("i{id}"),
                    class: SchedulingClass::Interactive,
                },
                0,
                now,
            );
        }
        let selected = (0..6)
            .filter_map(|_| policy.select(|_| true).map(|s| s.id))
            .collect::<Vec<_>>();
        assert!(selected.contains(&1));
        if selected.contains(&2) {
            assert!(
                selected.iter().position(|id| *id == 1) < selected.iter().position(|id| *id == 2)
            );
        }
    }

    #[test]
    fn weighted_deficits_preserve_class_shares_without_starvation() {
        let config = SchedulerPolicyConfig {
            interactive_weight: 4,
            standard_weight: 2,
            batch_weight: 1,
            promotion_rounds: 10_000,
            ..SchedulerPolicyConfig::default()
        };
        let mut policy = FairPolicy::new(config);
        let now = Instant::now();
        let flows = [
            FlowKey {
                principal: "interactive".into(),
                class: SchedulingClass::Interactive,
            },
            FlowKey {
                principal: "standard".into(),
                class: SchedulingClass::Standard,
            },
            FlowKey {
                principal: "batch".into(),
                class: SchedulingClass::Batch,
            },
        ];
        for (id, flow) in flows.iter().enumerate() {
            policy.enqueue(id as u64, flow.clone(), 0, now);
        }
        let mut counts = [0usize; 3];
        for _ in 0..70 {
            let selected = policy.select(|_| true).unwrap();
            let index = selected.id as usize;
            counts[index] += 1;
            policy.charge(&flows[index], 1, selected.charge_scale);
            policy.enqueue(selected.id, flows[index].clone(), 0, now);
        }
        assert_eq!(counts, [40, 20, 10]);
    }

    #[test]
    fn dropping_a_scheduled_session_cancels_its_shared_control_and_waiting_count() {
        let scheduler = WorkloadScheduler::new(
            Arc::new(DeterministicEngine),
            SchedulerPolicyConfig::default(),
        )
        .unwrap();
        let request = request(SchedulingClass::Standard, "principal", 1);
        let control = request.control.clone();
        let mut session = scheduler.start_session(request).unwrap();
        assert!(matches!(session.step().unwrap(), SessionStep::Token { .. }));
        assert_eq!(scheduler.status().metrics.waiting_for_consumer, 1);
        drop(session);
        assert!(matches!(control.check(), Err(Error::Cancelled)));
        for _ in 0..100 {
            if scheduler.status().metrics.waiting_for_consumer == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(scheduler.status().metrics.waiting_for_consumer, 0);
    }

    #[test]
    fn dropping_scheduler_cancels_and_joins_outstanding_work() {
        let scheduler = WorkloadScheduler::new(
            Arc::new(DeterministicEngine),
            SchedulerPolicyConfig::default(),
        )
        .unwrap();
        let request = request(SchedulingClass::Standard, "principal", 2);
        let control = request.control.clone();
        let mut session = scheduler.start_session(request).unwrap();
        drop(scheduler);
        assert!(matches!(control.check(), Err(Error::Cancelled)));
        assert!(matches!(session.step(), Err(Error::Cancelled)));
    }

    #[test]
    fn scheduler_interleaves_principals_and_emits_attributed_records() {
        let records = Arc::new(Mutex::new(Vec::new()));
        let captured = records.clone();
        let scheduler = WorkloadScheduler::new_with_diagnostics(
            Arc::new(DeterministicEngine),
            SchedulerPolicyConfig::default(),
            Some(move |line: &str| captured.lock().push(line.to_owned())),
        )
        .unwrap();
        let mut first = scheduler
            .start_session(request(SchedulingClass::Standard, "one", 2))
            .unwrap();
        let mut second = scheduler
            .start_session(request(SchedulingClass::Standard, "two", 2))
            .unwrap();
        assert!(matches!(first.step().unwrap(), SessionStep::Token { .. }));
        assert!(matches!(second.step().unwrap(), SessionStep::Token { .. }));
        first.finish().unwrap();
        second.finish().unwrap();
        assert!(scheduler.flush_diagnostics(Duration::from_secs(1)));
        let parsed = records
            .lock()
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert!(parsed.iter().any(|record| record["principal"] == "one"));
        assert!(parsed.iter().any(|record| record["principal"] == "two"));
        assert!(
            parsed
                .iter()
                .all(|record| record["execution_session_id"].is_string())
        );
    }

    #[test]
    fn diagnostic_overflow_is_counted_without_blocking_scheduler() {
        let scheduler = WorkloadScheduler::new_with_diagnostics(
            Arc::new(DeterministicEngine),
            SchedulerPolicyConfig {
                diagnostic_capacity: 1,
                ..SchedulerPolicyConfig::default()
            },
            Some(|_: &str| thread::sleep(Duration::from_millis(20))),
        )
        .unwrap();
        let mut sessions = (0..8)
            .map(|index| {
                scheduler
                    .start_session(request(SchedulingClass::Standard, &format!("p{index}"), 1))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for session in &mut sessions {
            let _ = session.step();
            let _ = session.finish();
        }
        assert!(scheduler.status().metrics.diagnostic_records_lost > 0);
        assert!(
            scheduler
                .status()
                .diagnostic_loss_by_kind
                .contains_key("scheduler_decision")
        );
    }

    #[test]
    fn distinct_model_slots_overlap_while_each_model_remains_serialized() {
        let engine = Arc::new(BlockingEngine::default());
        let scheduler =
            WorkloadScheduler::new(engine.clone(), SchedulerPolicyConfig::default()).unwrap();
        let mut sessions = [
            ("a", 1_u64, "a-one"),
            ("a", 1_u64, "a-two"),
            ("b", 2_u64, "b-one"),
        ]
        .into_iter()
        .map(|(model, epoch, principal)| {
            let mut request = request(SchedulingClass::Standard, principal, 1);
            request.model.id = model.into();
            request.model.epoch = epoch;
            scheduler.start_session(request).unwrap()
        })
        .collect::<Vec<_>>();
        let handles = sessions
            .drain(..)
            .map(|mut session| thread::spawn(move || session.step().unwrap()))
            .collect::<Vec<_>>();
        for handle in handles {
            assert!(matches!(
                handle.join().expect("client thread completes"),
                SessionStep::Finished(_)
            ));
        }
        assert_eq!(engine.max_active.load(Ordering::Acquire), 2);
        assert_eq!(engine.max_by_model.lock().get("a"), Some(&1));
        assert_eq!(engine.max_by_model.lock().get("b"), Some(&1));
    }
}
