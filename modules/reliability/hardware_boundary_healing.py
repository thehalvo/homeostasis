"""
Hardware/Software Boundary Healing for High-Reliability Systems.

This module provides healing capabilities at the hardware/software boundary,
including hardware fault detection, firmware recovery, and system-level healing.
"""

import asyncio
import json
import logging
import struct
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Types of hardware components."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK_CARD = "network_card"
    GPU = "gpu"
    POWER_SUPPLY = "power_supply"
    COOLING = "cooling"
    MOTHERBOARD = "motherboard"
    FIRMWARE = "firmware"
    SENSOR = "sensor"


class FaultType(Enum):
    """Types of hardware faults."""
    TEMPERATURE_HIGH = "temperature_high"
    TEMPERATURE_LOW = "temperature_low"
    VOLTAGE_UNSTABLE = "voltage_unstable"
    MEMORY_ERROR = "memory_error"
    DISK_FAILURE = "disk_failure"
    NETWORK_ERROR = "network_error"
    POWER_FAILURE = "power_failure"
    FAN_FAILURE = "fan_failure"
    FIRMWARE_CORRUPTION = "firmware_corruption"
    SENSOR_FAILURE = "sensor_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    INTERMITTENT_FAILURE = "intermittent_failure"


class HealingAction(Enum):
    """Types of healing actions."""
    THROTTLE_CPU = "throttle_cpu"
    MIGRATE_MEMORY = "migrate_memory"
    REMAP_DISK = "remap_disk"
    RESET_NETWORK = "reset_network"
    SWITCH_POWER_RAIL = "switch_power_rail"
    ADJUST_COOLING = "adjust_cooling"
    FIRMWARE_RECOVERY = "firmware_recovery"
    SENSOR_RECALIBRATION = "sensor_recalibration"
    WORKLOAD_MIGRATION = "workload_migration"
    COMPONENT_ISOLATION = "component_isolation"
    SYSTEM_REBOOT = "system_reboot"
    BIOS_RECOVERY = "bios_recovery"


@dataclass
class HardwareComponent:
    """Hardware component representation."""
    component_id: str
    component_type: HardwareType
    model: str
    serial_number: str
    location: str  # Physical location or slot
    status: str = "healthy"
    metrics: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    last_check: Optional[datetime] = None
    fault_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "model": self.model,
            "serial_number": self.serial_number,
            "location": self.location,
            "status": self.status,
            "metrics": self.metrics,
            "last_check": self.last_check.isoformat() if self.last_check else None
        }


@dataclass
class HardwareFault:
    """Hardware fault detection."""
    fault_id: str
    component: HardwareComponent
    fault_type: FaultType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    predicted_failure_time: Optional[timedelta] = None
    recommended_actions: List[HealingAction] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fault_id": self.fault_id,
            "component_id": self.component.component_id,
            "fault_type": self.fault_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "predicted_failure_time": self.predicted_failure_time.total_seconds() if self.predicted_failure_time else None,
            "recommended_actions": [a.value for a in self.recommended_actions]
        }


@dataclass
class HealingResult:
    """Result of hardware healing action."""
    action: HealingAction
    component: HardwareComponent
    success: bool
    timestamp: datetime
    duration: timedelta
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    error_message: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "component_id": self.component.component_id,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration.total_seconds(),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "error_message": self.error_message,
            "side_effects": self.side_effects
        }


class HardwareMonitor(ABC):
    """Abstract base class for hardware monitoring."""
    
    @abstractmethod
    async def get_metrics(self, component: HardwareComponent) -> Dict[str, float]:
        """Get current metrics for hardware component."""
        pass
    
    @abstractmethod
    async def detect_faults(self, component: HardwareComponent) -> List[HardwareFault]:
        """Detect hardware faults."""
        pass


class CPUMonitor(HardwareMonitor):
    """CPU monitoring and fault detection."""
    
    async def get_metrics(self, component: HardwareComponent) -> Dict[str, float]:
        """Get CPU metrics."""
        metrics = {}
        
        try:
            # Temperature monitoring
            temp_output = subprocess.check_output(["sensors", "-u"], text=True)
            for line in temp_output.split('\n'):
                if 'temp1_input' in line:
                    temp = float(line.split(':')[1].strip())
                    metrics['temperature'] = temp
                    break
            
            # Frequency monitoring
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        freq = float(line.split(':')[1].strip())
                        metrics['frequency'] = freq
                        break
            
            # Load monitoring
            with open('/proc/loadavg', 'r') as f:
                loads = f.read().split()
                metrics['load_1min'] = float(loads[0])
                metrics['load_5min'] = float(loads[1])
                metrics['load_15min'] = float(loads[2])
            
            # Error monitoring (MCE)
            mce_count = 0
            mce_path = Path('/sys/devices/system/machinecheck/machinecheck0/bank0/ce_count')
            if mce_path.exists():
                with open(mce_path, 'r') as f:
                    mce_count = int(f.read().strip())
            metrics['mce_errors'] = mce_count
            
        except Exception as e:
            logger.error(f"Failed to get CPU metrics: {e}")
        
        return metrics
    
    async def detect_faults(self, component: HardwareComponent) -> List[HardwareFault]:
        """Detect CPU faults."""
        faults = []
        metrics = await self.get_metrics(component)
        
        # Temperature check
        if 'temperature' in metrics:
            temp = metrics['temperature']
            min_temp, max_temp = component.thresholds.get('temperature', (0, 85))
            
            if temp > max_temp:
                faults.append(HardwareFault(
                    fault_id=f"cpu_temp_high_{int(time.time())}",
                    component=component,
                    fault_type=FaultType.TEMPERATURE_HIGH,
                    severity="high" if temp > max_temp + 10 else "medium",
                    timestamp=datetime.now(),
                    details={"temperature": temp, "threshold": max_temp},
                    recommended_actions=[HealingAction.THROTTLE_CPU, HealingAction.ADJUST_COOLING]
                ))
        
        # MCE error check
        if metrics.get('mce_errors', 0) > 0:
            faults.append(HardwareFault(
                fault_id=f"cpu_mce_{int(time.time())}",
                component=component,
                fault_type=FaultType.MEMORY_ERROR,
                severity="high",
                timestamp=datetime.now(),
                details={"mce_count": metrics['mce_errors']},
                recommended_actions=[HealingAction.WORKLOAD_MIGRATION]
            ))
        
        return faults


class MemoryMonitor(HardwareMonitor):
    """Memory monitoring and fault detection."""
    
    async def get_metrics(self, component: HardwareComponent) -> Dict[str, float]:
        """Get memory metrics."""
        metrics = {}
        
        try:
            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal:' in line:
                        metrics['total_mb'] = int(line.split()[1]) / 1024
                    elif 'MemAvailable:' in line:
                        metrics['available_mb'] = int(line.split()[1]) / 1024
                    elif 'MemFree:' in line:
                        metrics['free_mb'] = int(line.split()[1]) / 1024
            
            # ECC errors (if available)
            edac_path = Path('/sys/devices/system/edac/mc/mc0')
            if edac_path.exists():
                ce_count_path = edac_path / 'ce_count'
                ue_count_path = edac_path / 'ue_count'
                
                if ce_count_path.exists():
                    with open(ce_count_path, 'r') as f:
                        metrics['correctable_errors'] = int(f.read().strip())
                
                if ue_count_path.exists():
                    with open(ue_count_path, 'r') as f:
                        metrics['uncorrectable_errors'] = int(f.read().strip())
            
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
        
        return metrics
    
    async def detect_faults(self, component: HardwareComponent) -> List[HardwareFault]:
        """Detect memory faults."""
        faults = []
        metrics = await self.get_metrics(component)
        
        # ECC error check
        ce_errors = metrics.get('correctable_errors', 0)
        ue_errors = metrics.get('uncorrectable_errors', 0)
        
        if ue_errors > 0:
            faults.append(HardwareFault(
                fault_id=f"mem_ue_{int(time.time())}",
                component=component,
                fault_type=FaultType.MEMORY_ERROR,
                severity="critical",
                timestamp=datetime.now(),
                details={"uncorrectable_errors": ue_errors},
                recommended_actions=[HealingAction.MIGRATE_MEMORY, HealingAction.COMPONENT_ISOLATION]
            ))
        elif ce_errors > 100:  # Threshold for concern
            faults.append(HardwareFault(
                fault_id=f"mem_ce_{int(time.time())}",
                component=component,
                fault_type=FaultType.MEMORY_ERROR,
                severity="medium",
                timestamp=datetime.now(),
                details={"correctable_errors": ce_errors},
                predicted_failure_time=timedelta(days=7),
                recommended_actions=[HealingAction.MIGRATE_MEMORY]
            ))
        
        return faults


class DiskMonitor(HardwareMonitor):
    """Disk monitoring and fault detection using SMART."""
    
    async def get_metrics(self, component: HardwareComponent) -> Dict[str, float]:
        """Get disk metrics."""
        metrics = {}
        
        try:
            # SMART attributes
            smart_output = subprocess.check_output(
                ["smartctl", "-A", "-j", component.location],
                text=True
            )
            smart_data = json.loads(smart_output)
            
            # Extract key SMART attributes
            if 'ata_smart_attributes' in smart_data:
                for attr in smart_data['ata_smart_attributes']['table']:
                    if attr['id'] == 5:  # Reallocated sectors
                        metrics['reallocated_sectors'] = attr['raw']['value']
                    elif attr['id'] == 187:  # Reported uncorrectable
                        metrics['uncorrectable_errors'] = attr['raw']['value']
                    elif attr['id'] == 194:  # Temperature
                        metrics['temperature'] = attr['raw']['value']
                    elif attr['id'] == 9:  # Power on hours
                        metrics['power_on_hours'] = attr['raw']['value']
            
            # Disk usage
            df_output = subprocess.check_output(["df", "-B1", component.location], text=True)
            lines = df_output.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                metrics['total_bytes'] = int(parts[1])
                metrics['used_bytes'] = int(parts[2])
                metrics['usage_percent'] = float(parts[4].rstrip('%'))
            
        except Exception as e:
            logger.error(f"Failed to get disk metrics: {e}")
        
        return metrics
    
    async def detect_faults(self, component: HardwareComponent) -> List[HardwareFault]:
        """Detect disk faults."""
        faults = []
        metrics = await self.get_metrics(component)
        
        # Reallocated sectors check
        realloc = metrics.get('reallocated_sectors', 0)
        if realloc > 0:
            severity = "critical" if realloc > 100 else "high" if realloc > 10 else "medium"
            faults.append(HardwareFault(
                fault_id=f"disk_realloc_{int(time.time())}",
                component=component,
                fault_type=FaultType.DISK_FAILURE,
                severity=severity,
                timestamp=datetime.now(),
                details={"reallocated_sectors": realloc},
                predicted_failure_time=timedelta(days=30 if realloc < 10 else 7),
                recommended_actions=[HealingAction.REMAP_DISK, HealingAction.WORKLOAD_MIGRATION]
            ))
        
        # Temperature check
        temp = metrics.get('temperature', 0)
        if temp > 50:
            faults.append(HardwareFault(
                fault_id=f"disk_temp_{int(time.time())}",
                component=component,
                fault_type=FaultType.TEMPERATURE_HIGH,
                severity="medium",
                timestamp=datetime.now(),
                details={"temperature": temp},
                recommended_actions=[HealingAction.ADJUST_COOLING]
            ))
        
        return faults


class HardwareHealer:
    """Hardware healing coordinator."""
    
    def __init__(self):
        self.monitors: Dict[HardwareType, HardwareMonitor] = {
            HardwareType.CPU: CPUMonitor(),
            HardwareType.MEMORY: MemoryMonitor(),
            HardwareType.DISK: DiskMonitor()
        }
        self.components: Dict[str, HardwareComponent] = {}
        self.active_faults: Dict[str, HardwareFault] = {}
        self.healing_history: List[HealingResult] = []
    
    def register_component(self, component: HardwareComponent) -> None:
        """Register a hardware component for monitoring."""
        self.components[component.component_id] = component
        logger.info(f"Registered hardware component: {component.component_id}")
    
    async def monitor_components(self) -> List[HardwareFault]:
        """Monitor all registered components."""
        all_faults = []
        
        for component in self.components.values():
            monitor = self.monitors.get(component.component_type)
            if not monitor:
                continue
            
            try:
                # Get current metrics
                metrics = await monitor.get_metrics(component)
                component.metrics = metrics
                component.last_check = datetime.now()
                
                # Detect faults
                faults = await monitor.detect_faults(component)
                for fault in faults:
                    self.active_faults[fault.fault_id] = fault
                    all_faults.append(fault)
                
            except Exception as e:
                logger.error(f"Error monitoring component {component.component_id}: {e}")
        
        return all_faults
    
    async def heal_fault(
        self,
        fault: HardwareFault,
        action: HealingAction,
        force: bool = False
    ) -> HealingResult:
        """Execute healing action for a hardware fault."""
        component = fault.component
        start_time = datetime.now()
        metrics_before = component.metrics.copy()
        
        logger.info(f"Executing healing action {action.value} for {component.component_id}")
        
        try:
            # Validate action
            if not force and action not in fault.recommended_actions:
                logger.warning(f"Action {action.value} not recommended for fault {fault.fault_type.value}")
            
            # Execute healing action
            success = await self._execute_healing_action(component, action, fault)
            
            # Get metrics after healing
            monitor = self.monitors.get(component.component_type)
            if monitor:
                metrics_after = await monitor.get_metrics(component)
                component.metrics = metrics_after
            else:
                metrics_after = metrics_before
            
            result = HealingResult(
                action=action,
                component=component,
                success=success,
                timestamp=start_time,
                duration=datetime.now() - start_time,
                metrics_before=metrics_before,
                metrics_after=metrics_after
            )
            
            # Remove fault if healed
            if success and fault.fault_id in self.active_faults:
                del self.active_faults[fault.fault_id]
            
        except Exception as e:
            logger.error(f"Healing action failed: {e}")
            result = HealingResult(
                action=action,
                component=component,
                success=False,
                timestamp=start_time,
                duration=datetime.now() - start_time,
                metrics_before=metrics_before,
                metrics_after=metrics_before,
                error_message=str(e)
            )
        
        self.healing_history.append(result)
        return result
    
    async def _execute_healing_action(
        self,
        component: HardwareComponent,
        action: HealingAction,
        fault: HardwareFault
    ) -> bool:
        """Execute specific healing action."""
        if action == HealingAction.THROTTLE_CPU:
            return await self._throttle_cpu(component, fault)
        elif action == HealingAction.MIGRATE_MEMORY:
            return await self._migrate_memory(component, fault)
        elif action == HealingAction.REMAP_DISK:
            return await self._remap_disk(component, fault)
        elif action == HealingAction.ADJUST_COOLING:
            return await self._adjust_cooling(component, fault)
        elif action == HealingAction.FIRMWARE_RECOVERY:
            return await self._recover_firmware(component, fault)
        elif action == HealingAction.WORKLOAD_MIGRATION:
            return await self._migrate_workload(component, fault)
        elif action == HealingAction.COMPONENT_ISOLATION:
            return await self._isolate_component(component, fault)
        else:
            logger.warning(f"Unsupported healing action: {action.value}")
            return False
    
    async def _throttle_cpu(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Throttle CPU to reduce temperature."""
        try:
            # Set CPU frequency governor to powersave
            cpu_num = component.location.split('/')[-1]  # e.g., "cpu0"
            gov_path = f"/sys/devices/system/cpu/{cpu_num}/cpufreq/scaling_governor"
            
            with open(gov_path, 'w') as f:
                f.write("powersave")
            
            # Set maximum frequency to 80% of current
            max_freq_path = f"/sys/devices/system/cpu/{cpu_num}/cpufreq/scaling_max_freq"
            cur_freq_path = f"/sys/devices/system/cpu/{cpu_num}/cpufreq/scaling_cur_freq"
            
            with open(cur_freq_path, 'r') as f:
                current_freq = int(f.read().strip())
            
            new_max_freq = int(current_freq * 0.8)
            
            with open(max_freq_path, 'w') as f:
                f.write(str(new_max_freq))
            
            logger.info(f"Throttled CPU {cpu_num} to {new_max_freq} Hz")
            return True
            
        except Exception as e:
            logger.error(f"Failed to throttle CPU: {e}")
            return False
    
    async def _migrate_memory(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Migrate memory pages away from faulty region."""
        try:
            # This is a simplified example - real implementation would use
            # memory hotplug and page migration APIs
            
            # Mark memory region as offline
            memory_block = component.location  # e.g., "/sys/devices/system/memory/memory32"
            state_path = f"{memory_block}/state"
            
            with open(state_path, 'w') as f:
                f.write("offline")
            
            logger.info(f"Migrated memory from {memory_block}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate memory: {e}")
            return False
    
    async def _remap_disk(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Remap bad disk sectors."""
        try:
            # Force disk to remap bad sectors
            subprocess.run(
                ["hdparm", "--write-sector", "--yes-i-know-what-i-am-doing",
                 str(fault.details.get('bad_sector', 0)), component.location],
                check=True
            )
            
            logger.info(f"Remapped bad sectors on {component.location}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remap disk: {e}")
            return False
    
    async def _adjust_cooling(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Adjust cooling system."""
        try:
            # Increase fan speed
            # This is platform-specific - example for Dell servers
            subprocess.run(
                ["ipmitool", "raw", "0x30", "0x30", "0x01", "0xff"],
                check=True
            )
            
            logger.info("Increased fan speed to maximum")
            return True
            
        except Exception as e:
            logger.error(f"Failed to adjust cooling: {e}")
            return False
    
    async def _recover_firmware(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Recover corrupted firmware."""
        try:
            # Example: recover BIOS using flashrom
            backup_path = f"/var/lib/homeostasis/firmware/{component.serial_number}.rom"
            
            if Path(backup_path).exists():
                subprocess.run(
                    ["flashrom", "-p", "internal", "-w", backup_path],
                    check=True
                )
                logger.info(f"Recovered firmware from {backup_path}")
                return True
            else:
                logger.error("No firmware backup available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to recover firmware: {e}")
            return False
    
    async def _migrate_workload(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Migrate workload to healthy hardware."""
        try:
            # This would integrate with container orchestration or VM management
            # Example: drain Kubernetes node
            node_name = component.metadata.get('node_name')
            if node_name:
                subprocess.run(
                    ["kubectl", "drain", node_name, "--ignore-daemonsets", "--delete-emptydir-data"],
                    check=True
                )
                logger.info(f"Drained workloads from node {node_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to migrate workload: {e}")
            return False
    
    async def _isolate_component(self, component: HardwareComponent, fault: HardwareFault) -> bool:
        """Isolate faulty component from system."""
        try:
            # Mark component as isolated
            component.status = "isolated"
            
            # Remove from active use
            if component.component_type == HardwareType.CPU:
                # Take CPU offline
                cpu_path = f"/sys/devices/system/cpu/{component.location}/online"
                with open(cpu_path, 'w') as f:
                    f.write("0")
            elif component.component_type == HardwareType.MEMORY:
                # Already handled in migrate_memory
                pass
            elif component.component_type == HardwareType.DISK:
                # Remove from RAID array or mark as failed
                subprocess.run(
                    ["mdadm", "--fail", "/dev/md0", component.location],
                    check=False
                )
            
            logger.info(f"Isolated component {component.component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to isolate component: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system hardware health."""
        healthy_components = [c for c in self.components.values() if c.status == "healthy"]
        degraded_components = [c for c in self.components.values() if c.status == "degraded"]
        failed_components = [c for c in self.components.values() if c.status in ["failed", "isolated"]]
        
        return {
            "total_components": len(self.components),
            "healthy": len(healthy_components),
            "degraded": len(degraded_components),
            "failed": len(failed_components),
            "active_faults": len(self.active_faults),
            "critical_faults": len([f for f in self.active_faults.values() if f.severity == "critical"]),
            "components": [c.to_dict() for c in self.components.values()],
            "recent_healings": [h.to_dict() for h in self.healing_history[-10:]]
        }


# Example hardware configurations
def create_server_hardware() -> List[HardwareComponent]:
    """Create hardware components for a typical server."""
    return [
        HardwareComponent(
            component_id="cpu0",
            component_type=HardwareType.CPU,
            model="Intel Xeon E5-2680 v4",
            serial_number="CPU001",
            location="/sys/devices/system/cpu/cpu0",
            thresholds={"temperature": (10, 85), "frequency": (1200, 3300)},
            capabilities=["throttling", "turbo", "c-states"]
        ),
        HardwareComponent(
            component_id="memory0",
            component_type=HardwareType.MEMORY,
            model="Samsung DDR4-2400 32GB",
            serial_number="MEM001",
            location="/sys/devices/system/memory/memory0",
            thresholds={"correctable_errors": (0, 100)},
            capabilities=["ecc", "hotplug"]
        ),
        HardwareComponent(
            component_id="disk0",
            component_type=HardwareType.DISK,
            model="Samsung SSD 860 EVO 1TB",
            serial_number="DISK001",
            location="/dev/sda",
            thresholds={"temperature": (0, 50), "reallocated_sectors": (0, 10)},
            capabilities=["smart", "trim", "secure-erase"]
        )
    ]


async def monitor_and_heal_hardware():
    """Example of hardware monitoring and healing."""
    healer = HardwareHealer()
    
    # Register components
    for component in create_server_hardware():
        healer.register_component(component)
    
    # Monitor continuously
    while True:
        try:
            # Check for faults
            faults = await healer.monitor_components()
            
            # Heal critical faults automatically
            for fault in faults:
                if fault.severity == "critical" and fault.recommended_actions:
                    # Execute first recommended action
                    action = fault.recommended_actions[0]
                    result = await healer.heal_fault(fault, action)
                    
                    if result.success:
                        logger.info(f"Successfully healed {fault.fault_type.value} on {fault.component.component_id}")
                    else:
                        logger.error(f"Failed to heal {fault.fault_type.value}: {result.error_message}")
            
            # Get system health
            health = healer.get_system_health()
            logger.info(f"System health: {health['healthy']} healthy, {health['degraded']} degraded, {health['failed']} failed")
            
            # Wait before next check
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in hardware monitoring: {e}")
            await asyncio.sleep(30)