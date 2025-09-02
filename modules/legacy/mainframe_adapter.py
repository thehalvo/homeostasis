"""
Mainframe environment adapter for Homeostasis.

This module provides integration with mainframe systems including
z/OS, z/VM, z/VSE, and AS/400 (IBM i) environments.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class MainframeType(Enum):
    """Supported mainframe platforms."""
    ZOS = "z/OS"
    ZVM = "z/VM"
    ZVSE = "z/VSE"
    AS400 = "AS/400"
    UNISYS = "Unisys"
    TANDEM = "Tandem"


class JobStatus(Enum):
    """JCL job execution statuses."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABENDED = "ABENDED"


@dataclass
class MainframeError:
    """Represents a mainframe system error."""
    job_name: str
    job_id: str
    step_name: Optional[str]
    abend_code: Optional[str]
    error_message: str
    timestamp: datetime
    dataset: Optional[str]
    program: Optional[str]
    mainframe_type: MainframeType
    severity: str
    sysout: Optional[str]


@dataclass
class MainframeEnvironment:
    """Mainframe environment configuration."""
    host: str
    port: int
    mainframe_type: MainframeType
    lpar: Optional[str]
    subsystem: Optional[str]
    codepage: str = "EBCDIC"
    timeout: int = 300
    batch_class: str = "A"
    message_class: str = "X"


class MainframeAdapter:
    """
    Adapter for integrating with mainframe environments.
    
    Provides error detection, job monitoring, and healing capabilities
    for mainframe batch jobs and online transactions.
    """
    
    def __init__(self, environment: MainframeEnvironment):
        self.environment = environment
        self.connection = None
        self._abend_codes = self._load_abend_codes()
        self._jcl_patterns = self._compile_jcl_patterns()
        
    def _load_abend_codes(self) -> Dict[str, Dict[str, Any]]:
        """Load mainframe ABEND code definitions."""
        return {
            # System ABENDs
            "S0C1": {
                "description": "Operation exception",
                "severity": "high",
                "category": "program_error",
                "healing_strategy": "check_instruction_validity"
            },
            "S0C4": {
                "description": "Protection exception - invalid memory access",
                "severity": "high",
                "category": "memory_error",
                "healing_strategy": "check_storage_boundaries"
            },
            "S0C7": {
                "description": "Data exception - invalid decimal data",
                "severity": "high",
                "category": "data_error",
                "healing_strategy": "validate_packed_decimal"
            },
            "S013": {
                "description": "Open error on dataset",
                "severity": "medium",
                "category": "io_error",
                "healing_strategy": "check_dataset_allocation"
            },
            "S037": {
                "description": "Dataset space allocation failure",
                "severity": "medium",
                "category": "space_error",
                "healing_strategy": "increase_space_allocation"
            },
            "S106": {
                "description": "LINK or LOAD module not found",
                "severity": "high",
                "category": "linkage_error",
                "healing_strategy": "check_load_library"
            },
            "S222": {
                "description": "Job cancelled by operator",
                "severity": "low",
                "category": "operational",
                "healing_strategy": "check_job_dependencies"
            },
            "S322": {
                "description": "Job exceeded CPU time limit",
                "severity": "medium",
                "category": "resource_error",
                "healing_strategy": "increase_time_limit"
            },
            "S722": {
                "description": "Output limit exceeded",
                "severity": "medium",
                "category": "resource_error",
                "healing_strategy": "increase_output_limit"
            },
            "S806": {
                "description": "Module not found in JOBLIB/STEPLIB",
                "severity": "high",
                "category": "linkage_error",
                "healing_strategy": "check_library_concatenation"
            },
            "S878": {
                "description": "Insufficient virtual storage",
                "severity": "high",
                "category": "memory_error",
                "healing_strategy": "increase_region_size"
            },
            "S913": {
                "description": "Security violation",
                "severity": "high",
                "category": "security_error",
                "healing_strategy": "check_dataset_permissions"
            },
            # User ABENDs
            "U0001": {
                "description": "User-defined error",
                "severity": "medium",
                "category": "application_error",
                "healing_strategy": "check_application_logic"
            },
            "U4038": {
                "description": "LE runtime error",
                "severity": "high",
                "category": "runtime_error",
                "healing_strategy": "check_le_options"
            }
        }
        
    def _compile_jcl_patterns(self) -> Dict[str, re.Pattern]:
        """Compile JCL parsing patterns."""
        return {
            "job_card": re.compile(r"^//(\w+)\s+JOB\s+(.*)$"),
            "exec_card": re.compile(r"^//(\w+)\s+EXEC\s+(.*)$"),
            "dd_card": re.compile(r"^//(\w+)\s+DD\s+(.*)$"),
            "proc_card": re.compile(r"^//(\w+)\s+PROC\s*(.*)$"),
            "pend_card": re.compile(r"^//\s+PEND\s*$"),
            "if_card": re.compile(r"^//\s+IF\s+(.*)$"),
            "endif_card": re.compile(r"^//\s+ENDIF\s*$"),
            "abend_msg": re.compile(r"IEF\d+I\s+(\w+)\s+.*ABEND=(\w+)"),
            "completion_code": re.compile(r"COMPLETION CODE - SYSTEM=(\w+).*USER=(\w+)")
        }
        
    def connect(self) -> bool:
        """Establish connection to mainframe."""
        try:
            # In a real implementation, this would connect via:
            # - FTP for file transfer
            # - TN3270 for terminal emulation
            # - z/OSMF REST APIs for modern interfaces
            # - MQ Series for messaging
            logger.info(f"Connecting to {self.environment.mainframe_type.value} "
                       f"at {self.environment.host}:{self.environment.port}")
            
            # Placeholder for actual connection logic
            self.connection = {
                "connected": True,
                "session_id": f"SESS{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "lpar": self.environment.lpar,
                "subsystem": self.environment.subsystem
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to mainframe: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from mainframe."""
        if self.connection:
            logger.info("Disconnecting from mainframe")
            self.connection = None
            
    def parse_job_log(self, job_output: str) -> List[MainframeError]:
        """Parse job output for errors and ABENDs."""
        errors = []
        
        lines = job_output.split('\n')
        current_job = None
        current_step = None
        
        for line in lines:
            # Parse job card
            job_match = self._jcl_patterns["job_card"].match(line)
            if job_match:
                current_job = job_match.group(1)
                continue
                
            # Parse exec card
            exec_match = self._jcl_patterns["exec_card"].match(line)
            if exec_match:
                current_step = exec_match.group(1)
                continue
                
            # Check for ABEND messages
            abend_match = self._jcl_patterns["abend_msg"].search(line)
            if abend_match:
                step_name = abend_match.group(1)
                abend_code = abend_match.group(2)
                
                error = MainframeError(
                    job_name=current_job or "UNKNOWN",
                    job_id=self._extract_job_id(job_output),
                    step_name=step_name,
                    abend_code=abend_code,
                    error_message=line.strip(),
                    timestamp=datetime.now(),
                    dataset=None,
                    program=self._extract_program_name(job_output, step_name),
                    mainframe_type=self.environment.mainframe_type,
                    severity=self._get_abend_severity(abend_code),
                    sysout=job_output
                )
                errors.append(error)
                
            # Check for completion codes
            cc_match = self._jcl_patterns["completion_code"].search(line)
            if cc_match:
                system_code = cc_match.group(1)
                user_code = cc_match.group(2)
                
                if system_code != "000" or user_code != "0000":
                    error = MainframeError(
                        job_name=current_job or "UNKNOWN",
                        job_id=self._extract_job_id(job_output),
                        step_name=current_step,
                        abend_code=f"S{system_code}" if system_code != "000" else f"U{user_code}",
                        error_message=f"Job failed with completion code SYSTEM={system_code} USER={user_code}",
                        timestamp=datetime.now(),
                        dataset=None,
                        program=self._extract_program_name(job_output, current_step),
                        mainframe_type=self.environment.mainframe_type,
                        severity="high" if system_code != "000" else "medium",
                        sysout=job_output
                    )
                    errors.append(error)
                    
        return errors
        
    def _extract_job_id(self, job_output: str) -> str:
        """Extract job ID from job output."""
        # Look for JOBnnnnn pattern
        match = re.search(r"(JOB\d{5})", job_output)
        return match.group(1) if match else "UNKNOWN"
        
    def _extract_program_name(self, job_output: str, step_name: str) -> Optional[str]:
        """Extract program name for a given step."""
        if not step_name:
            return None
            
        # Look for PGM= in the EXEC statement
        pattern = f"//{step_name}\\s+EXEC\\s+PGM=([\\w\\.]+)"
        match = re.search(pattern, job_output)
        return match.group(1) if match else None
        
    def _get_abend_severity(self, abend_code: str) -> str:
        """Get severity level for ABEND code."""
        if abend_code in self._abend_codes:
            return self._abend_codes[abend_code]["severity"]
        return "medium"  # Default severity
        
    def analyze_error(self, error: MainframeError) -> Dict[str, Any]:
        """Analyze mainframe error and suggest healing strategy."""
        analysis = {
            "error": error,
            "root_cause": None,
            "healing_strategies": [],
            "risk_level": "medium",
            "automated_fix_available": False
        }
        
        if error.abend_code and error.abend_code in self._abend_codes:
            abend_info = self._abend_codes[error.abend_code]
            analysis["root_cause"] = abend_info["description"]
            
            # Determine healing strategies based on ABEND type
            strategy = abend_info["healing_strategy"]
            
            if strategy == "check_instruction_validity":
                analysis["healing_strategies"] = [
                    {
                        "action": "recompile_program",
                        "description": "Recompile program with optimization disabled",
                        "automated": True
                    },
                    {
                        "action": "check_compile_options",
                        "description": "Verify compiler options and listings",
                        "automated": False
                    }
                ]
                
            elif strategy == "check_storage_boundaries":
                analysis["healing_strategies"] = [
                    {
                        "action": "increase_region_size",
                        "description": "Increase REGION parameter in JCL",
                        "automated": True,
                        "parameters": {"region": "0M"}
                    },
                    {
                        "action": "check_array_bounds",
                        "description": "Review array and table definitions",
                        "automated": False
                    }
                ]
                
            elif strategy == "validate_packed_decimal":
                analysis["healing_strategies"] = [
                    {
                        "action": "initialize_numeric_fields",
                        "description": "Add initialization for numeric fields",
                        "automated": True
                    },
                    {
                        "action": "add_numeric_validation",
                        "description": "Add NUMPROC(NOPFD) compiler option",
                        "automated": True
                    }
                ]
                
            elif strategy == "check_dataset_allocation":
                analysis["healing_strategies"] = [
                    {
                        "action": "preallocate_dataset",
                        "description": "Add dataset pre-allocation step",
                        "automated": True
                    },
                    {
                        "action": "check_dataset_attributes",
                        "description": "Verify DCB parameters match program expectations",
                        "automated": False
                    }
                ]
                
            elif strategy == "increase_space_allocation":
                analysis["healing_strategies"] = [
                    {
                        "action": "increase_primary_space",
                        "description": "Increase primary space allocation",
                        "automated": True,
                        "parameters": {"space": "(CYL,(100,50),RLSE)"}
                    },
                    {
                        "action": "add_secondary_extents",
                        "description": "Add secondary extent allocation",
                        "automated": True
                    }
                ]
                
            # Set automated fix availability
            analysis["automated_fix_available"] = any(
                s["automated"] for s in analysis["healing_strategies"]
            )
            
        return analysis
        
    def generate_jcl_patch(self, error: MainframeError, strategy: Dict[str, Any]) -> Optional[str]:
        """Generate JCL patch for automated healing."""
        if not strategy.get("automated", False):
            return None
            
        action = strategy["action"]
        
        if action == "increase_region_size":
            region = strategy.get("parameters", {}).get("region", "0M")
            return f"""//* Healing patch for ABEND {error.abend_code}
//* Increasing region size to resolve storage issues
//{error.step_name}  EXEC PGM={error.program},REGION={region}
"""
            
        elif action == "increase_primary_space":
            space = strategy.get("parameters", {}).get("space", "(CYL,(100,50),RLSE)")
            return f"""//* Healing patch for ABEND {error.abend_code}
//* Increasing dataset space allocation
//SYSIN    DD  *
  ALTER '{error.dataset}' SPACE{space}
/*
"""
            
        elif action == "preallocate_dataset":
            return f"""//* Healing patch for ABEND {error.abend_code}
//ALLOC    EXEC PGM=IEFBR14
//DD1      DD  DSN={error.dataset},
//             DISP=(NEW,CATLG,DELETE),
//             SPACE=(CYL,(10,5),RLSE),
//             DCB=(RECFM=FB,LRECL=80,BLKSIZE=27920)
"""
            
        elif action == "add_numeric_validation":
            return f"""//* Healing patch for ABEND {error.abend_code}
//* Adding NUMPROC option to handle invalid packed decimal
//COBOL.PARM='NUMPROC(NOPFD)'
"""
            
        return None
        
    def submit_job(self, jcl: str) -> Tuple[str, JobStatus]:
        """Submit JCL job to mainframe."""
        if not self.connection:
            raise RuntimeError("Not connected to mainframe")
            
        # In a real implementation, this would:
        # 1. Submit JCL via FTP or z/OSMF REST API
        # 2. Monitor job execution
        # 3. Retrieve job output
        
        job_id = f"JOB{datetime.now().strftime('%H%M%S')[-5:]}"
        logger.info(f"Submitting job {job_id}")
        
        # Simulate job submission
        return job_id, JobStatus.PENDING
        
    def monitor_job(self, job_id: str) -> JobStatus:
        """Monitor job execution status."""
        if not self.connection:
            raise RuntimeError("Not connected to mainframe")
            
        # In a real implementation, this would query job status
        # via SDSF or z/OSMF REST API
        
        logger.info(f"Monitoring job {job_id}")
        return JobStatus.COMPLETED
        
    def retrieve_job_output(self, job_id: str) -> str:
        """Retrieve job output from spool."""
        if not self.connection:
            raise RuntimeError("Not connected to mainframe")
            
        # In a real implementation, this would retrieve
        # job output from JES spool
        
        logger.info(f"Retrieving output for job {job_id}")
        
        # Return sample output for demonstration
        return f"""1                       J E S 2  J O B  L O G
 
 JOB{job_id}  JOB  USER01    
 
IEF236I ALLOC. FOR JOB{job_id} STEP01
IEF237I JES2 ALLOCATED TO SYSIN
IEF237I JES2 ALLOCATED TO SYSPRINT
IEF142I JOB{job_id} STEP01 - STEP WAS EXECUTED - COND CODE 0000
IEF285I   USER01.JOB{job_id}.D0000101.?        SYSOUT
IEF373I STEP/STEP01  /START 2024001.1230
IEF374I STEP/STEP01  /STOP  2024001.1230 CPU    0MIN 00.12SEC
IEF375I  JOB/JOB{job_id}/START 2024001.1230
IEF376I  JOB/JOB{job_id}/STOP  2024001.1230 CPU    0MIN 00.12SEC
"""
        
    def get_healing_recommendations(self, error: MainframeError) -> List[Dict[str, Any]]:
        """Get healing recommendations for mainframe errors."""
        recommendations = []
        
        # Analyze the error
        analysis = self.analyze_error(error)
        
        for strategy in analysis["healing_strategies"]:
            recommendation = {
                "title": strategy["action"].replace("_", " ").title(),
                "description": strategy["description"],
                "automated": strategy["automated"],
                "risk_level": "low" if strategy["automated"] else "medium",
                "implementation": None
            }
            
            if strategy["automated"]:
                # Generate automated patch
                patch = self.generate_jcl_patch(error, strategy)
                if patch:
                    recommendation["implementation"] = {
                        "type": "jcl_patch",
                        "content": patch
                    }
                    
            recommendations.append(recommendation)
            
        # Add general recommendations
        if error.abend_code and error.abend_code.startswith("S0C"):
            recommendations.append({
                "title": "Enable Diagnostic Options",
                "description": "Compile with TEST option and generate dump for analysis",
                "automated": False,
                "risk_level": "low",
                "implementation": {
                    "type": "compile_option",
                    "content": "CBL SSRANGE,TEST(ALL),DUMP"
                }
            })
            
        return recommendations
        
    def apply_healing(self, error: MainframeError, recommendation: Dict[str, Any]) -> bool:
        """Apply healing recommendation to mainframe job."""
        if not recommendation.get("automated", False):
            logger.warning("Manual intervention required for this healing strategy")
            return False
            
        implementation = recommendation.get("implementation")
        if not implementation:
            logger.error("No implementation available for recommendation")
            return False
            
        try:
            if implementation["type"] == "jcl_patch":
                # Submit healing JCL
                job_id, status = self.submit_job(implementation["content"])
                
                # Wait for completion
                final_status = self.monitor_job(job_id)
                
                if final_status == JobStatus.COMPLETED:
                    logger.info(f"Healing job {job_id} completed successfully")
                    return True
                else:
                    logger.error(f"Healing job {job_id} failed with status {final_status}")
                    return False
                    
            elif implementation["type"] == "compile_option":
                # This would require recompilation of the program
                logger.info("Recompilation required with options: " + implementation["content"])
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply healing: {e}")
            return False
            
        return False