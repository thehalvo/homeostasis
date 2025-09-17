"""
Distributed transaction healing for polyglot microservices.
Manages transaction recovery across services written in different languages.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .microservice_healer import ServiceError
from .unified_error_taxonomy import UnifiedErrorTaxonomy


class TransactionState(Enum):
    """States of a distributed transaction."""

    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"


class TransactionPattern(Enum):
    """Distributed transaction patterns."""

    TWO_PHASE_COMMIT = "2pc"
    SAGA = "saga"
    TRY_CONFIRM_CANCEL = "tcc"
    EVENTUAL_CONSISTENCY = "eventual"
    OUTBOX = "outbox"


class CompensationStrategy(Enum):
    """Strategies for compensating failed transactions."""

    ROLLBACK = "rollback"
    COMPENSATE = "compensate"
    RETRY = "retry"
    MANUAL = "manual"
    IGNORE = "ignore"


@dataclass
class TransactionParticipant:
    """A participant in a distributed transaction."""

    service_id: str
    service_name: str
    language: str
    endpoint: str
    state: TransactionState
    last_update: datetime
    error: Optional[ServiceError] = None
    compensation_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTransaction:
    """Represents a distributed transaction across services."""

    transaction_id: str
    pattern: TransactionPattern
    participants: List[TransactionParticipant]
    state: TransactionState
    created_at: datetime
    updated_at: datetime
    initiator_service: str
    business_operation: str
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    metadata: Dict[str, Any] = field(default_factory=dict)
    compensation_chain: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TransactionRecoveryPlan:
    """Plan for recovering a failed distributed transaction."""

    transaction_id: str
    strategy: CompensationStrategy
    affected_participants: List[TransactionParticipant]
    recovery_steps: List[Dict[str, Any]]
    estimated_recovery_time: timedelta
    risk_assessment: Dict[str, Any]
    rollback_data: Dict[str, Any] = field(default_factory=dict)


class TransactionCoordinator(ABC):
    """Abstract base class for transaction coordination strategies."""

    @abstractmethod
    async def prepare(self, transaction: DistributedTransaction) -> bool:
        """Prepare phase of transaction."""
        pass

    @abstractmethod
    async def commit(self, transaction: DistributedTransaction) -> bool:
        """Commit phase of transaction."""
        pass

    @abstractmethod
    async def abort(self, transaction: DistributedTransaction) -> bool:
        """Abort/rollback transaction."""
        pass

    @abstractmethod
    async def get_recovery_plan(
        self, transaction: DistributedTransaction
    ) -> TransactionRecoveryPlan:
        """Generate recovery plan for failed transaction."""
        pass


class TwoPhaseCommitCoordinator(TransactionCoordinator):
    """Implements 2PC protocol for distributed transactions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def prepare(self, transaction: DistributedTransaction) -> bool:
        """Execute prepare phase across all participants."""
        transaction.state = TransactionState.PREPARING
        all_prepared = True

        # Send prepare to all participants
        prepare_tasks = []
        for participant in transaction.participants:
            task = self._prepare_participant(participant, transaction)
            prepare_tasks.append(task)

        results = await asyncio.gather(*prepare_tasks, return_exceptions=True)

        # Check results
        for i, result in enumerate(results):
            participant = transaction.participants[i]
            if isinstance(result, Exception) or not result:
                participant.state = TransactionState.FAILED
                participant.error = ServiceError(
                    service_id=participant.service_id,
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    error_type="PrepareFailure",
                    message=(
                        str(result)
                        if isinstance(result, Exception)
                        else "Prepare failed"
                    ),
                    stack_trace=None,
                    language=participant.language,
                )
                all_prepared = False
            else:
                participant.state = TransactionState.PREPARED

        transaction.state = (
            TransactionState.PREPARED if all_prepared else TransactionState.FAILED
        )
        return all_prepared

    async def commit(self, transaction: DistributedTransaction) -> bool:
        """Execute commit phase if all participants prepared."""
        if transaction.state != TransactionState.PREPARED:
            return False

        transaction.state = TransactionState.COMMITTING
        all_committed = True

        # Send commit to all participants
        commit_tasks = []
        for participant in transaction.participants:
            if participant.state == TransactionState.PREPARED:
                task = self._commit_participant(participant, transaction)
                commit_tasks.append(task)

        results = await asyncio.gather(*commit_tasks, return_exceptions=True)

        # Check results
        for i, result in enumerate(results):
            participant = transaction.participants[i]
            if isinstance(result, Exception) or not result:
                participant.state = TransactionState.FAILED
                all_committed = False
            else:
                participant.state = TransactionState.COMMITTED

        transaction.state = (
            TransactionState.COMMITTED if all_committed else TransactionState.FAILED
        )
        return all_committed

    async def abort(self, transaction: DistributedTransaction) -> bool:
        """Abort transaction and rollback all participants."""
        transaction.state = TransactionState.ABORTING
        all_aborted = True

        # Send abort to all participants
        abort_tasks = []
        for participant in transaction.participants:
            if participant.state in [
                TransactionState.PREPARED,
                TransactionState.COMMITTED,
            ]:
                task = self._abort_participant(participant, transaction)
                abort_tasks.append(task)

        results = await asyncio.gather(*abort_tasks, return_exceptions=True)

        # Check results
        for i, result in enumerate(results):
            if isinstance(result, Exception) or not result:
                all_aborted = False

        transaction.state = (
            TransactionState.ABORTED if all_aborted else TransactionState.FAILED
        )
        return all_aborted

    async def get_recovery_plan(
        self, transaction: DistributedTransaction
    ) -> TransactionRecoveryPlan:
        """Generate recovery plan for 2PC failure."""
        affected = [
            p for p in transaction.participants if p.state != TransactionState.COMMITTED
        ]

        recovery_steps: List[Dict[str, Any]] = []
        if transaction.state == TransactionState.PREPARING:
            # Failed during prepare - abort all
            recovery_steps.append(
                {
                    "action": "abort_all",
                    "description": "Abort all participants that started preparing",
                }
            )
            strategy = CompensationStrategy.ROLLBACK

        elif transaction.state == TransactionState.COMMITTING:
            # Failed during commit - complex recovery needed
            committed = [
                p
                for p in transaction.participants
                if p.state == TransactionState.COMMITTED
            ]
            if committed:
                # Some committed - need compensation
                recovery_steps.append(
                    {
                        "action": "compensate_committed",
                        "participants": [p.service_id for p in committed],
                        "description": "Compensate already committed participants",
                    }
                )
                strategy = CompensationStrategy.COMPENSATE
            else:
                # None committed - can rollback
                recovery_steps.append(
                    {
                        "action": "rollback_all",
                        "description": "Rollback all prepared participants",
                    }
                )
                strategy = CompensationStrategy.ROLLBACK

        else:
            strategy = CompensationStrategy.MANUAL

        return TransactionRecoveryPlan(
            transaction_id=transaction.transaction_id,
            strategy=strategy,
            affected_participants=affected,
            recovery_steps=recovery_steps,
            estimated_recovery_time=timedelta(seconds=len(affected) * 5),
            risk_assessment={
                "data_inconsistency_risk": (
                    len(committed) > 0 if "committed" in locals() else False
                ),
                "recovery_complexity": (
                    "high" if strategy == CompensationStrategy.COMPENSATE else "medium"
                ),
            },
        )

    async def _prepare_participant(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Send prepare request to participant."""
        # This would make actual API call to participant
        self.logger.info(f"Preparing participant: {participant.service_name}")
        await asyncio.sleep(0.1)  # Simulate network call
        return True

    async def _commit_participant(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Send commit request to participant."""
        self.logger.info(f"Committing participant: {participant.service_name}")
        await asyncio.sleep(0.1)  # Simulate network call
        return True

    async def _abort_participant(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Send abort request to participant."""
        self.logger.info(f"Aborting participant: {participant.service_name}")
        await asyncio.sleep(0.1)  # Simulate network call
        return True


class SagaCoordinator(TransactionCoordinator):
    """Implements Saga pattern for long-running transactions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def prepare(self, transaction: DistributedTransaction) -> bool:
        """Saga doesn't have explicit prepare phase."""
        return True

    async def commit(self, transaction: DistributedTransaction) -> bool:
        """Execute saga steps sequentially."""
        transaction.state = TransactionState.COMMITTING

        for i, participant in enumerate(transaction.participants):
            try:
                # Execute step
                success = await self._execute_saga_step(participant, transaction)

                if success:
                    participant.state = TransactionState.COMMITTED

                    # Record compensation info
                    transaction.compensation_chain.append(
                        {
                            "participant": participant.service_id,
                            "compensation_endpoint": participant.compensation_endpoint,
                            "order": i,
                            "timestamp": datetime.now(),
                        }
                    )
                else:
                    participant.state = TransactionState.FAILED
                    transaction.state = TransactionState.FAILED

                    # Trigger compensation
                    await self._compensate_saga(transaction, i - 1)
                    return False

            except Exception as e:
                self.logger.error(f"Saga step failed: {e}")
                participant.state = TransactionState.FAILED
                transaction.state = TransactionState.FAILED

                # Trigger compensation
                await self._compensate_saga(transaction, i - 1)
                return False

        transaction.state = TransactionState.COMMITTED
        return True

    async def abort(self, transaction: DistributedTransaction) -> bool:
        """Compensate all executed steps."""
        return await self._compensate_saga(
            transaction, len(transaction.participants) - 1
        )

    async def get_recovery_plan(
        self, transaction: DistributedTransaction
    ) -> TransactionRecoveryPlan:
        """Generate recovery plan for saga failure."""
        # Find where it failed
        failed_index = -1
        for i, participant in enumerate(transaction.participants):
            if participant.state == TransactionState.FAILED:
                failed_index = i
                break

        recovery_steps: List[Dict[str, Any]] = []

        # Compensate in reverse order
        for i in range(failed_index - 1, -1, -1):
            participant = transaction.participants[i]
            if participant.state == TransactionState.COMMITTED:
                recovery_steps.append(
                    {
                        "action": "compensate",
                        "participant": participant.service_id,
                        "endpoint": participant.compensation_endpoint,
                        "order": len(recovery_steps),
                    }
                )

        return TransactionRecoveryPlan(
            transaction_id=transaction.transaction_id,
            strategy=CompensationStrategy.COMPENSATE,
            affected_participants=transaction.participants[:failed_index],
            recovery_steps=recovery_steps,
            estimated_recovery_time=timedelta(seconds=len(recovery_steps) * 3),
            risk_assessment={
                "compensation_complexity": "medium",
                "data_consistency_risk": "low",  # Saga designed for this
            },
        )

    async def _execute_saga_step(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Execute a single saga step."""
        self.logger.info(f"Executing saga step: {participant.service_name}")
        await asyncio.sleep(0.1)  # Simulate execution
        return True

    async def _compensate_saga(
        self, transaction: DistributedTransaction, up_to_index: int
    ) -> bool:
        """Compensate saga steps in reverse order."""
        transaction.state = TransactionState.COMPENSATING

        for i in range(up_to_index, -1, -1):
            participant = transaction.participants[i]
            if participant.state == TransactionState.COMMITTED:
                try:
                    success = await self._compensate_participant(participant)
                    if success:
                        participant.state = TransactionState.COMPENSATED
                    else:
                        self.logger.error(
                            f"Failed to compensate: {participant.service_name}"
                        )
                        return False
                except Exception as e:
                    self.logger.error(f"Compensation error: {e}")
                    return False

        transaction.state = TransactionState.COMPENSATED
        return True

    async def _compensate_participant(
        self, participant: TransactionParticipant
    ) -> bool:
        """Execute compensation for a participant."""
        if not participant.compensation_endpoint:
            self.logger.warning(
                f"No compensation endpoint for: {participant.service_name}"
            )
            return False

        self.logger.info(f"Compensating: {participant.service_name}")
        await asyncio.sleep(0.1)  # Simulate compensation
        return True


class DistributedTransactionHealer:
    """
    Manages distributed transaction healing across polyglot microservices.
    Detects and recovers from distributed transaction failures.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.coordinators: Dict[TransactionPattern, TransactionCoordinator] = {
            TransactionPattern.TWO_PHASE_COMMIT: TwoPhaseCommitCoordinator(config),
            TransactionPattern.SAGA: SagaCoordinator(config),
        }
        self.active_transactions: Dict[str, DistributedTransaction] = {}
        self.transaction_history: List[DistributedTransaction] = []
        self.unified_taxonomy = UnifiedErrorTaxonomy()

    async def create_transaction(
        self,
        pattern: TransactionPattern,
        participants: List[Dict[str, Any]],
        business_operation: str,
        initiator_service: str,
        timeout: Optional[timedelta] = None,
    ) -> DistributedTransaction:
        """Create a new distributed transaction."""
        transaction_id = str(uuid.uuid4())

        # Create participant objects
        participant_objs = []
        for p in participants:
            participant = TransactionParticipant(
                service_id=p["service_id"],
                service_name=p["service_name"],
                language=p["language"],
                endpoint=p["endpoint"],
                state=TransactionState.PENDING,
                last_update=datetime.now(),
                compensation_endpoint=p.get("compensation_endpoint"),
                metadata=p.get("metadata", {}),
            )
            participant_objs.append(participant)

        transaction = DistributedTransaction(
            transaction_id=transaction_id,
            pattern=pattern,
            participants=participant_objs,
            state=TransactionState.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            initiator_service=initiator_service,
            business_operation=business_operation,
            timeout=timeout or timedelta(seconds=30),
        )

        self.active_transactions[transaction_id] = transaction
        self.logger.info(
            f"Created {pattern.value} transaction {transaction_id} "
            f"for {business_operation} with {len(participants)} participants"
        )

        return transaction

    async def execute_transaction(self, transaction: DistributedTransaction) -> bool:
        """Execute a distributed transaction."""
        coordinator = self.coordinators.get(transaction.pattern)
        if not coordinator:
            raise ValueError(f"No coordinator for pattern: {transaction.pattern}")

        try:
            # Start transaction monitoring
            monitor_task = asyncio.create_task(self._monitor_transaction(transaction))

            # Execute based on pattern
            if transaction.pattern == TransactionPattern.TWO_PHASE_COMMIT:
                # Prepare phase
                prepared = await coordinator.prepare(transaction)
                if not prepared:
                    await coordinator.abort(transaction)
                    return False

                # Commit phase
                committed = await coordinator.commit(transaction)
                if not committed:
                    # Try to recover
                    recovery_plan = await coordinator.get_recovery_plan(transaction)
                    await self.execute_recovery(recovery_plan, transaction)
                    return False

            elif transaction.pattern == TransactionPattern.SAGA:
                # Execute saga
                success = await coordinator.commit(transaction)
                if not success:
                    return False

            # Cancel monitoring
            monitor_task.cancel()

            # Move to history
            transaction.updated_at = datetime.now()
            self.transaction_history.append(transaction)
            del self.active_transactions[transaction.transaction_id]

            return True

        except Exception as e:
            self.logger.error(f"Transaction execution failed: {e}")
            transaction.state = TransactionState.FAILED

            # Get recovery plan
            recovery_plan = await coordinator.get_recovery_plan(transaction)
            await self.execute_recovery(recovery_plan, transaction)

            return False

    async def _monitor_transaction(self, transaction: DistributedTransaction) -> None:
        """Monitor transaction for timeouts and failures."""
        start_time = datetime.now()

        while transaction.state not in [
            TransactionState.COMMITTED,
            TransactionState.ABORTED,
            TransactionState.COMPENSATED,
            TransactionState.FAILED,
        ]:
            # Check timeout
            if datetime.now() - start_time > transaction.timeout:
                self.logger.warning(
                    f"Transaction {transaction.transaction_id} timed out"
                )
                transaction.state = TransactionState.FAILED
                break

            # Check participant health
            for participant in transaction.participants:
                if participant.error:
                    self.logger.warning(
                        f"Participant {participant.service_name} reported error"
                    )

            await asyncio.sleep(1)

    async def detect_transaction_anomalies(
        self, transaction: DistributedTransaction
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in distributed transaction."""
        anomalies = []

        # Check for stuck transactions
        if transaction.state in [
            TransactionState.PREPARING,
            TransactionState.COMMITTING,
        ]:
            duration = datetime.now() - transaction.updated_at
            if duration > timedelta(seconds=10):
                anomalies.append(
                    {
                        "type": "stuck_transaction",
                        "duration": duration.total_seconds(),
                        "state": transaction.state.value,
                    }
                )

        # Check for inconsistent participant states
        participant_states = {p.state for p in transaction.participants}
        if len(participant_states) > 2:
            anomalies.append(
                {
                    "type": "inconsistent_states",
                    "states": [s.value for s in participant_states],
                }
            )

        # Check for language-specific issues
        language_errors: Dict[str, List[ErrorContext]] = {}
        for participant in transaction.participants:
            if participant.error:
                lang = participant.language
                if lang not in language_errors:
                    language_errors[lang] = []
                language_errors[lang].append(participant.error)

        if language_errors:
            anomalies.append(
                {
                    "type": "language_specific_errors",
                    "errors_by_language": {
                        lang: len(errors) for lang, errors in language_errors.items()
                    },
                }
            )

        return anomalies

    async def execute_recovery(
        self,
        recovery_plan: TransactionRecoveryPlan,
        transaction: DistributedTransaction,
    ) -> bool:
        """Execute a transaction recovery plan."""
        self.logger.info(
            f"Executing recovery plan for transaction {transaction.transaction_id} "
            f"using strategy: {recovery_plan.strategy.value}"
        )

        success = True

        for step in recovery_plan.recovery_steps:
            action = step.get("action")

            if action == "abort_all":
                coordinator = self.coordinators[transaction.pattern]
                success = await coordinator.abort(transaction)

            elif action == "compensate_committed":
                # Execute compensation for specific participants
                participants = step.get("participants", [])
                for p_id in participants:
                    participant = next(
                        (p for p in transaction.participants if p.service_id == p_id),
                        None,
                    )
                    if participant:
                        comp_success = await self._execute_compensation(
                            participant, transaction
                        )
                        success = success and comp_success

            elif action == "rollback_all":
                # Rollback all prepared participants
                for participant in transaction.participants:
                    if participant.state == TransactionState.PREPARED:
                        rb_success = await self._rollback_participant(
                            participant, transaction
                        )
                        success = success and rb_success

        return success

    async def _execute_compensation(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Execute compensation for a participant."""
        if not participant.compensation_endpoint:
            self.logger.error(
                f"No compensation endpoint for {participant.service_name}"
            )
            return False

        self.logger.info(
            f"Compensating {participant.service_name} at {participant.compensation_endpoint}"
        )

        # This would make actual API call
        await asyncio.sleep(0.1)  # Simulate call
        participant.state = TransactionState.COMPENSATED

        return True

    async def _rollback_participant(
        self, participant: TransactionParticipant, transaction: DistributedTransaction
    ) -> bool:
        """Rollback a participant."""
        self.logger.info(f"Rolling back {participant.service_name}")

        # This would make actual rollback API call
        await asyncio.sleep(0.1)  # Simulate call
        participant.state = TransactionState.ABORTED

        return True

    async def generate_healing_recommendations(
        self, failed_transaction: DistributedTransaction
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for preventing similar failures."""
        recommendations = []

        # Analyze failure patterns
        anomalies = await self.detect_transaction_anomalies(failed_transaction)

        for anomaly in anomalies:
            if anomaly["type"] == "stuck_transaction":
                recommendations.append(
                    {
                        "type": "timeout_adjustment",
                        "description": "Increase transaction timeout",
                        "current_timeout": failed_transaction.timeout.total_seconds(),
                        "suggested_timeout": failed_transaction.timeout.total_seconds()
                        * 2,
                    }
                )

            elif anomaly["type"] == "inconsistent_states":
                recommendations.append(
                    {
                        "type": "pattern_change",
                        "description": "Consider using Saga pattern instead of 2PC",
                        "reason": "Better handling of partial failures",
                    }
                )

        # Language-specific recommendations
        for participant in failed_transaction.participants:
            if participant.error:
                # Classify error
                error_data = {
                    "type": participant.error.error_type,
                    "message": participant.error.message,
                }

                unified_error = self.unified_taxonomy.classify_error(
                    error_data, participant.language
                )

                # Get fix recommendations
                fixes = self.unified_taxonomy.get_fix_recommendations(
                    unified_error, participant.language
                )

                for fix in fixes:
                    recommendations.append(
                        {
                            "type": "code_fix",
                            "service": participant.service_name,
                            "language": participant.language,
                            "fix": fix["strategy"],
                            "confidence": fix["confidence"],
                        }
                    )

        # Transaction pattern recommendations
        if failed_transaction.pattern == TransactionPattern.TWO_PHASE_COMMIT:
            if len(failed_transaction.participants) > 5:
                recommendations.append(
                    {
                        "type": "architecture",
                        "description": "Consider event sourcing for large transactions",
                        "reason": "2PC doesn't scale well with many participants",
                    }
                )

        return recommendations

    async def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get metrics about distributed transactions."""
        total_transactions = len(self.transaction_history)

        if total_transactions == 0:
            return {
                "total_transactions": 0,
                "success_rate": 0,
                "patterns_used": {},
                "average_participants": 0,
                "languages_involved": [],
            }

        successful = sum(
            1 for t in self.transaction_history if t.state == TransactionState.COMMITTED
        )

        patterns_used: Dict[str, int] = {}
        languages: Set[str] = set()
        total_participants = 0

        for transaction in self.transaction_history:
            pattern = transaction.pattern.value
            patterns_used[pattern] = patterns_used.get(pattern, 0) + 1

            total_participants += len(transaction.participants)

            for participant in transaction.participants:
                languages.add(participant.language)

        return {
            "total_transactions": total_transactions,
            "success_rate": successful / total_transactions,
            "active_transactions": len(self.active_transactions),
            "patterns_used": patterns_used,
            "average_participants": total_participants / total_transactions,
            "languages_involved": list(languages),
            "failure_reasons": self._analyze_failure_reasons(),
        }

    def _analyze_failure_reasons(self) -> Dict[str, int]:
        """Analyze common failure reasons."""
        reasons = {}

        for transaction in self.transaction_history:
            if transaction.state == TransactionState.FAILED:
                for participant in transaction.participants:
                    if participant.error:
                        reason = participant.error.error_type
                        reasons[reason] = reasons.get(reason, 0) + 1

        return reasons
