"""
Polyglot stack trace analysis for distributed systems.
Analyzes and correlates stack traces across different programming languages.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .unified_error_taxonomy import UnifiedErrorTaxonomy


class FrameType(Enum):
    """Type of stack frame."""

    APPLICATION = "application"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    SYSTEM = "system"
    NATIVE = "native"


@dataclass
class StackFrame:
    """Represents a single frame in a stack trace."""

    index: int
    file_path: Optional[str]
    line_number: Optional[int]
    function_name: str
    module_name: Optional[str]
    class_name: Optional[str]
    frame_type: FrameType
    language: str
    raw_frame: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StackTrace:
    """Represents a complete stack trace."""

    trace_id: str
    language: str
    error_message: str
    error_type: str
    frames: List[StackFrame]
    timestamp: datetime
    service_name: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelatedStackTrace:
    """Stack traces correlated across services."""

    correlation_id: str
    root_trace: StackTrace
    related_traces: List[StackTrace]
    causality_chain: List[Tuple[str, str]]  # (from_service, to_service)
    total_services: int
    languages_involved: Set[str]
    root_cause_frame: Optional[StackFrame] = None


class StackTraceParser(ABC):
    """Abstract base class for language-specific stack trace parsers."""

    @abstractmethod
    def parse(self, raw_trace: str) -> StackTrace:
        """Parse a raw stack trace into structured format."""
        pass

    @abstractmethod
    def identify_language(self, raw_trace: str) -> bool:
        """Check if this parser can handle the given trace."""
        pass


class PolyglotStackTraceAnalyzer:
    """
    Analyzes stack traces across multiple programming languages.
    Provides correlation and root cause analysis for distributed errors.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.parsers: Dict[str, StackTraceParser] = {}
        self.unified_taxonomy = UnifiedErrorTaxonomy()

        # Initialize built-in parsers
        self._initialize_parsers()

    def _initialize_parsers(self):
        """Initialize language-specific parsers."""
        self.register_parser("python", PythonStackTraceParser())
        self.register_parser("javascript", JavaScriptStackTraceParser())
        self.register_parser("java", JavaStackTraceParser())
        self.register_parser("go", GoStackTraceParser())
        self.register_parser("csharp", CSharpStackTraceParser())
        self.register_parser("ruby", RubyStackTraceParser())
        self.register_parser("php", PHPStackTraceParser())
        self.register_parser("rust", RustStackTraceParser())

    def register_parser(self, language: str, parser: StackTraceParser) -> None:
        """Register a stack trace parser for a language."""
        self.parsers[language] = parser
        self.logger.info(f"Registered {language} stack trace parser")

    def parse_stack_trace(
        self, raw_trace: str, language: Optional[str] = None
    ) -> StackTrace:
        """
        Parse a raw stack trace into structured format.
        Auto-detects language if not specified.
        """
        if language:
            if language in self.parsers:
                return self.parsers[language].parse(raw_trace)
            else:
                raise ValueError(f"No parser available for language: {language}")

        # Auto-detect language
        for lang, parser in self.parsers.items():
            if parser.identify_language(raw_trace):
                self.logger.info(f"Auto-detected language: {lang}")
                return parser.parse(raw_trace)

        raise ValueError("Could not identify stack trace language")

    def correlate_traces(
        self,
        traces: List[StackTrace],
        correlation_hints: Optional[Dict[str, Any]] = None,
    ) -> List[CorrelatedStackTrace]:
        """
        Correlate stack traces from different services.
        Uses timestamps, error messages, and correlation hints.
        """
        correlated_groups = []
        processed = set()

        for i, trace in enumerate(traces):
            if i in processed:
                continue

            # Find related traces
            related = []
            for j, other_trace in enumerate(traces):
                if i != j and j not in processed:
                    if self._are_traces_related(trace, other_trace, correlation_hints):
                        related.append(other_trace)
                        processed.add(j)

            if related:
                # Create correlated trace group
                correlation = self._create_correlation(trace, related)
                correlated_groups.append(correlation)

        return correlated_groups

    def _are_traces_related(
        self,
        trace1: StackTrace,
        trace2: StackTrace,
        hints: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Determine if two traces are related."""
        # Check temporal proximity (within 5 seconds)
        time_diff = abs((trace1.timestamp - trace2.timestamp).total_seconds())
        if time_diff > 5:
            return False

        # Check for common identifiers in metadata
        if hints:
            trace_id = hints.get("trace_id")
            if trace_id:
                return (
                    trace1.metadata.get("trace_id") == trace_id
                    and trace2.metadata.get("trace_id") == trace_id
                )

        # Check for error propagation patterns
        # e.g., HTTP client error in one trace, server error in another
        if self._check_error_propagation(trace1, trace2):
            return True

        # Check for common stack frames (shared libraries)
        common_frames = self._find_common_frames(trace1, trace2)
        if len(common_frames) > 0:
            return True

        return False

    def _check_error_propagation(self, trace1: StackTrace, trace2: StackTrace) -> bool:
        """Check if errors show propagation pattern."""
        # Network errors often propagate
        network_errors = [
            "connection",
            "timeout",
            "refused",
            "reset",
            "broken pipe",
            "network",
            "socket",
        ]

        trace1_network = any(
            err in trace1.error_message.lower() for err in network_errors
        )
        trace2_network = any(
            err in trace2.error_message.lower() for err in network_errors
        )

        return trace1_network and trace2_network

    def _find_common_frames(
        self, trace1: StackTrace, trace2: StackTrace
    ) -> List[Tuple[StackFrame, StackFrame]]:
        """Find common frames between traces."""
        common = []

        for frame1 in trace1.frames:
            for frame2 in trace2.frames:
                if (
                    frame1.function_name == frame2.function_name
                    and frame1.module_name == frame2.module_name
                ):
                    common.append((frame1, frame2))

        return common

    def _create_correlation(
        self, root_trace: StackTrace, related_traces: List[StackTrace]
    ) -> CorrelatedStackTrace:
        """Create a correlated stack trace group."""
        all_traces = [root_trace] + related_traces

        # Determine causality chain
        causality_chain = self._determine_causality_chain(all_traces)

        # Find root cause frame
        root_cause_frame = self._find_root_cause_frame(root_trace)

        # Collect all languages
        languages = {trace.language for trace in all_traces}

        return CorrelatedStackTrace(
            correlation_id=f"corr_{datetime.now().timestamp()}",
            root_trace=root_trace,
            related_traces=related_traces,
            causality_chain=causality_chain,
            total_services=len(all_traces),
            languages_involved=languages,
            root_cause_frame=root_cause_frame,
        )

    def _determine_causality_chain(
        self, traces: List[StackTrace]
    ) -> List[Tuple[str, str]]:
        """Determine the causality chain between services."""
        chain = []

        # Sort by timestamp
        sorted_traces = sorted(traces, key=lambda t: t.timestamp)

        for i in range(len(sorted_traces) - 1):
            from_service = sorted_traces[i].service_name or f"service_{i}"
            to_service = sorted_traces[i + 1].service_name or f"service_{i + 1}"
            chain.append((from_service, to_service))

        return chain

    def _find_root_cause_frame(self, trace: StackTrace) -> Optional[StackFrame]:
        """Find the most likely root cause frame in a trace."""
        # Look for the first application frame (not library/framework)
        for frame in trace.frames:
            if frame.frame_type == FrameType.APPLICATION:
                return frame

        # If no application frame, return the topmost frame
        return trace.frames[0] if trace.frames else None

    def analyze_trace_patterns(self, trace: StackTrace) -> Dict[str, Any]:
        """
        Analyze patterns in a stack trace.
        Identifies common issues like infinite recursion, deep calls, etc.
        """
        analysis = {
            "trace_id": trace.trace_id,
            "depth": len(trace.frames),
            "patterns": [],
            "anomalies": [],
            "recommendations": [],
        }

        # Check for recursion
        recursion = self._detect_recursion(trace)
        if recursion:
            analysis["patterns"].append({"type": "recursion", "details": recursion})
            analysis["recommendations"].append(
                "Add recursion depth limit or base case check"
            )

        # Check for deep call stacks
        if len(trace.frames) > 50:
            analysis["anomalies"].append(
                {"type": "deep_stack", "depth": len(trace.frames)}
            )
            analysis["recommendations"].append(
                "Consider refactoring to reduce call depth"
            )

        # Check for framework overhead
        framework_ratio = self._calculate_framework_ratio(trace)
        if framework_ratio > 0.7:
            analysis["anomalies"].append(
                {"type": "high_framework_overhead", "ratio": framework_ratio}
            )

        # Language-specific patterns
        lang_patterns = self._analyze_language_patterns(trace)
        analysis["patterns"].extend(lang_patterns)

        return analysis

    def _detect_recursion(self, trace: StackTrace) -> Optional[Dict[str, Any]]:
        """Detect recursion patterns in stack trace."""
        function_counts = {}

        for frame in trace.frames:
            key = f"{frame.module_name}.{frame.function_name}"
            function_counts[key] = function_counts.get(key, 0) + 1

        # Find functions that appear multiple times
        recursive_funcs = {
            func: count for func, count in function_counts.items() if count > 2
        }

        if recursive_funcs:
            max_func = max(recursive_funcs, key=recursive_funcs.get)
            return {
                "function": max_func,
                "occurrences": recursive_funcs[max_func],
                "all_recursive": recursive_funcs,
            }

        return None

    def _calculate_framework_ratio(self, trace: StackTrace) -> float:
        """Calculate ratio of framework/library frames to total."""
        if not trace.frames:
            return 0.0

        framework_frames = sum(
            1
            for frame in trace.frames
            if frame.frame_type in [FrameType.FRAMEWORK, FrameType.LIBRARY]
        )

        return framework_frames / len(trace.frames)

    def _analyze_language_patterns(self, trace: StackTrace) -> List[Dict[str, Any]]:
        """Analyze language-specific patterns."""
        patterns = []

        if trace.language == "python":
            # Check for common Python patterns
            if "maximum recursion depth exceeded" in trace.error_message:
                patterns.append(
                    {
                        "type": "python_recursion_limit",
                        "description": "Hit Python recursion limit",
                    }
                )

        elif trace.language == "javascript":
            # Check for async/promise issues
            if "UnhandledPromiseRejection" in trace.error_type:
                patterns.append(
                    {
                        "type": "unhandled_promise",
                        "description": "Unhandled promise rejection",
                    }
                )

        elif trace.language == "java":
            # Check for thread issues
            thread_keywords = ["Thread", "Runnable", "Executor"]
            if any(kw in str(trace.frames) for kw in thread_keywords):
                patterns.append(
                    {
                        "type": "java_threading",
                        "description": "Threading-related stack trace",
                    }
                )

        return patterns

    def generate_healing_suggestions(
        self, correlated_trace: CorrelatedStackTrace
    ) -> List[Dict[str, Any]]:
        """
        Generate healing suggestions for correlated traces.
        Considers all languages involved.
        """
        suggestions = []

        # Analyze root cause
        if correlated_trace.root_cause_frame:
            frame = correlated_trace.root_cause_frame

            # Get unified error classification
            error_data = {
                "type": correlated_trace.root_trace.error_type,
                "message": correlated_trace.root_trace.error_message,
            }

            unified_error = self.unified_taxonomy.classify_error(
                error_data, frame.language
            )

            # Get fix recommendations
            for lang in correlated_trace.languages_involved:
                fixes = self.unified_taxonomy.get_fix_recommendations(
                    unified_error, lang
                )

                for fix in fixes:
                    suggestions.append(
                        {
                            "language": lang,
                            "frame": f"{frame.file_path}:{frame.line_number}",
                            "function": frame.function_name,
                            "suggestion": fix["strategy"],
                            "confidence": fix["confidence"],
                            "implementation": fix.get("language_specific", {}),
                        }
                    )

        # Add distributed system suggestions
        if len(correlated_trace.related_traces) > 0:
            suggestions.append(
                {
                    "type": "distributed_system",
                    "suggestion": "Add distributed tracing for better correlation",
                    "languages": list(correlated_trace.languages_involved),
                }
            )

            suggestions.append(
                {
                    "type": "circuit_breaker",
                    "suggestion": "Implement circuit breakers between services",
                    "services": [
                        trace.service_name for trace in correlated_trace.related_traces
                    ],
                }
            )

        return suggestions

    def export_analysis(
        self, correlated_trace: CorrelatedStackTrace, format: str = "json"
    ) -> str:
        """Export stack trace analysis in various formats."""
        if format == "json":
            data = {
                "correlation_id": correlated_trace.correlation_id,
                "root_trace": self._trace_to_dict(correlated_trace.root_trace),
                "related_traces": [
                    self._trace_to_dict(trace)
                    for trace in correlated_trace.related_traces
                ],
                "causality_chain": correlated_trace.causality_chain,
                "languages": list(correlated_trace.languages_involved),
                "root_cause": (
                    self._frame_to_dict(correlated_trace.root_cause_frame)
                    if correlated_trace.root_cause_frame
                    else None
                ),
            }
            return json.dumps(data, indent=2, default=str)

        elif format == "text":
            lines = [
                "Correlated Stack Trace Analysis",
                "================================",
                f"Correlation ID: {correlated_trace.correlation_id}",
                f"Services Involved: {correlated_trace.total_services}",
                f"Languages: {', '.join(correlated_trace.languages_involved)}",
                "\nCausality Chain:",
            ]

            for from_svc, to_svc in correlated_trace.causality_chain:
                lines.append(f"  {from_svc} -> {to_svc}")

            lines.append("\nRoot Cause:")
            if correlated_trace.root_cause_frame:
                frame = correlated_trace.root_cause_frame
                lines.append(
                    f"  {frame.file_path}:{frame.line_number} "
                    f"in {frame.function_name}"
                )

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _trace_to_dict(self, trace: StackTrace) -> Dict[str, Any]:
        """Convert stack trace to dictionary."""
        return {
            "trace_id": trace.trace_id,
            "language": trace.language,
            "error_type": trace.error_type,
            "error_message": trace.error_message,
            "service_name": trace.service_name,
            "timestamp": trace.timestamp.isoformat(),
            "frames": [self._frame_to_dict(f) for f in trace.frames],
        }

    def _frame_to_dict(self, frame: Optional[StackFrame]) -> Optional[Dict[str, Any]]:
        """Convert stack frame to dictionary."""
        if not frame:
            return None

        return {
            "index": frame.index,
            "file": frame.file_path,
            "line": frame.line_number,
            "function": frame.function_name,
            "module": frame.module_name,
            "class": frame.class_name,
            "type": frame.frame_type.value,
            "language": frame.language,
        }


# Language-specific parser implementations


class PythonStackTraceParser(StackTraceParser):
    """Parser for Python stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        python_indicators = [
            "Traceback (most recent call last)",
            'File "<stdin>"',
            'File "<module>"',
            '.py", line',
            "NameError:",
            "TypeError:",
            "ValueError:",
            "AttributeError:",
            "KeyError:",
            "IndexError:",
        ]
        return any(indicator in raw_trace for indicator in python_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "Exception"

        frame_pattern = re.compile(r'File "([^"]+)", line (\d+), in (.+)')

        for i, line in enumerate(lines):
            line = line.strip()

            # Parse frame
            match = frame_pattern.match(line)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                function_name = match.group(3)

                # Determine frame type
                frame_type = FrameType.APPLICATION
                if "site-packages" in file_path or "lib/python" in file_path:
                    frame_type = FrameType.LIBRARY
                elif any(fw in file_path for fw in ["django", "flask", "fastapi"]):
                    frame_type = FrameType.FRAMEWORK

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    module_name=self._extract_module_from_path(file_path),
                    class_name=None,  # Would need more parsing
                    frame_type=frame_type,
                    language="python",
                    raw_frame=line,
                )
                frames.append(frame)

            # Parse error message
            elif ":" in line and not line.startswith("File"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    error_type = parts[0]
                    error_message = parts[1].strip()

        return StackTrace(
            trace_id=f"python_{datetime.now().timestamp()}",
            language="python",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )

    def _extract_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path."""
        # Simple extraction - could be enhanced
        parts = file_path.split("/")
        if parts[-1].endswith(".py"):
            return parts[-1][:-3]
        return parts[-1]


class JavaScriptStackTraceParser(StackTraceParser):
    """Parser for JavaScript/Node.js stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        js_indicators = [
            "at Object.<anonymous>",
            "at Module._compile",
            "node_modules",
            ".js:",
            "TypeError:",
            "ReferenceError:",
            "SyntaxError:",
            "at processTicksAndRejections",
        ]
        return any(indicator in raw_trace for indicator in js_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "Error"

        # First line often contains error type and message
        if lines:
            first_line = lines[0]
            if ":" in first_line:
                parts = first_line.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else ""

        # Parse frames
        frame_pattern = re.compile(r"at\s+(?:([^\s]+)\s+)?\(?(.*?):(\d+):(\d+)\)?")

        for line in lines[1:]:
            match = frame_pattern.search(line)
            if match:
                function_name = match.group(1) or "anonymous"
                file_path = match.group(2)
                line_number = int(match.group(3))

                # Determine frame type
                frame_type = FrameType.APPLICATION
                if "node_modules" in file_path:
                    frame_type = FrameType.LIBRARY
                elif any(fw in file_path for fw in ["express", "react", "vue"]):
                    frame_type = FrameType.FRAMEWORK
                elif file_path.startswith("node:"):
                    frame_type = FrameType.SYSTEM

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    module_name=self._extract_module_from_path(file_path),
                    class_name=None,
                    frame_type=frame_type,
                    language="javascript",
                    raw_frame=line.strip(),
                )
                frames.append(frame)

        return StackTrace(
            trace_id=f"javascript_{datetime.now().timestamp()}",
            language="javascript",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )

    def _extract_module_from_path(self, file_path: str) -> str:
        if "node_modules" in file_path:
            parts = file_path.split("node_modules/")
            if len(parts) > 1:
                module_parts = parts[1].split("/")
                return module_parts[0]
        return file_path.split("/")[-1]


class JavaStackTraceParser(StackTraceParser):
    """Parser for Java stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        java_indicators = [
            "at java.",
            "at com.",
            "at org.",
            ".java:",
            "Exception in thread",
            "Caused by:",
            "NullPointerException",
            "ClassNotFoundException",
        ]
        return any(indicator in raw_trace for indicator in java_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "Exception"

        # Parse exception info
        if lines:
            first_line = lines[0]
            if "Exception" in first_line or "Error" in first_line:
                parts = first_line.split(":")
                if "thread" in first_line:
                    # Format: Exception in thread "main" java.lang.NullPointerException: message
                    error_parts = parts[1].strip().split(" ", 1)
                    error_type = error_parts[0]
                    error_message = error_parts[1] if len(error_parts) > 1 else ""
                else:
                    error_type = parts[0].strip()
                    error_message = parts[1].strip() if len(parts) > 1 else ""

        # Parse frames
        frame_pattern = re.compile(r"at\s+([\w.$]+)\(([\w.]+):(\d+)\)")

        for line in lines:
            match = frame_pattern.search(line)
            if match:
                full_method = match.group(1)
                file_name = match.group(2)
                line_number = int(match.group(3))

                # Split class and method
                parts = full_method.rsplit(".", 1)
                class_name = parts[0] if len(parts) > 1 else ""
                method_name = parts[1] if len(parts) > 1 else full_method

                # Determine frame type
                frame_type = FrameType.APPLICATION
                if any(pkg in class_name for pkg in ["java.", "javax.", "sun."]):
                    frame_type = FrameType.SYSTEM
                elif any(
                    fw in class_name
                    for fw in ["springframework", "apache", "hibernate"]
                ):
                    frame_type = FrameType.FRAMEWORK

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_name,
                    line_number=line_number,
                    function_name=method_name,
                    module_name=(
                        class_name.split(".")[0] if "." in class_name else class_name
                    ),
                    class_name=class_name,
                    frame_type=frame_type,
                    language="java",
                    raw_frame=line.strip(),
                )
                frames.append(frame)

        return StackTrace(
            trace_id=f"java_{datetime.now().timestamp()}",
            language="java",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )


class GoStackTraceParser(StackTraceParser):
    """Parser for Go stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        go_indicators = [
            "goroutine",
            "panic:",
            ".go:",
            "runtime.panic",
            "main.main()",
            "/usr/local/go/src/",
        ]
        return any(indicator in raw_trace for indicator in go_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "panic"

        # Parse panic message
        for line in lines:
            if line.startswith("panic:"):
                error_message = line[6:].strip()
                break

        # Parse frames (Go shows function then file:line)
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for function name pattern
            if "(" in line and ")" in line and not line.startswith("/"):
                function_line = line

                # Next line should have file:line
                if i + 1 < len(lines):
                    file_line = lines[i + 1].strip()
                    file_match = re.match(r"(.+\.go):(\d+)", file_line)

                    if file_match:
                        file_path = file_match.group(1)
                        line_number = int(file_match.group(2))

                        # Extract function name
                        func_match = re.match(r"([\w./]+)\(", function_line)
                        function_name = (
                            func_match.group(1) if func_match else function_line
                        )

                        # Determine frame type
                        frame_type = FrameType.APPLICATION
                        if "/usr/local/go/src/" in file_path:
                            frame_type = FrameType.SYSTEM
                        elif "vendor/" in file_path or "pkg/mod/" in file_path:
                            frame_type = FrameType.LIBRARY

                        frame = StackFrame(
                            index=len(frames),
                            file_path=file_path,
                            line_number=line_number,
                            function_name=function_name,
                            module_name=self._extract_module_from_func(function_name),
                            class_name=None,
                            frame_type=frame_type,
                            language="go",
                            raw_frame=f"{function_line}\n{file_line}",
                        )
                        frames.append(frame)

                i += 2
            else:
                i += 1

        return StackTrace(
            trace_id=f"go_{datetime.now().timestamp()}",
            language="go",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )

    def _extract_module_from_func(self, function_name: str) -> str:
        """Extract module from Go function name."""
        parts = function_name.split("/")
        if len(parts) > 1:
            return parts[-2]
        return function_name.split(".")[0]


class CSharpStackTraceParser(StackTraceParser):
    """Parser for C# stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        csharp_indicators = [
            "at System.",
            "at Microsoft.",
            ".cs:line",
            "NullReferenceException",
            "InvalidOperationException",
            "in <filename>:line",
        ]
        return any(indicator in raw_trace for indicator in csharp_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "Exception"

        # Parse exception info
        if lines:
            first_line = lines[0]
            if "Exception" in first_line:
                parts = first_line.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else ""

        # Parse frames
        frame_pattern = re.compile(r"at\s+([\w.<>]+)\s+in\s+(.+):line\s+(\d+)")

        for line in lines:
            match = frame_pattern.search(line)
            if match:
                method_info = match.group(1)
                file_path = match.group(2)
                line_number = int(match.group(3))

                # Extract class and method
                parts = method_info.rsplit(".", 1)
                class_name = parts[0] if len(parts) > 1 else ""
                method_name = parts[1] if len(parts) > 1 else method_info

                # Determine frame type
                frame_type = FrameType.APPLICATION
                if any(ns in class_name for ns in ["System.", "Microsoft."]):
                    frame_type = FrameType.SYSTEM

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=method_name,
                    module_name=(
                        class_name.split(".")[0] if "." in class_name else class_name
                    ),
                    class_name=class_name,
                    frame_type=frame_type,
                    language="csharp",
                    raw_frame=line.strip(),
                )
                frames.append(frame)

        return StackTrace(
            trace_id=f"csharp_{datetime.now().timestamp()}",
            language="csharp",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )


class RubyStackTraceParser(StackTraceParser):
    """Parser for Ruby stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        ruby_indicators = [
            ".rb:",
            "from /",
            "`block in",
            "`rescue in",
            "NoMethodError",
            "NameError",
            "`<main>'",
        ]
        return any(indicator in raw_trace for indicator in ruby_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "StandardError"

        # Parse error info
        if lines:
            first_line = lines[0]
            if ":" in first_line and ".rb:" not in first_line:
                error_message = first_line
                if "Error" in first_line:
                    error_type = first_line.split(":")[0].strip()

        # Parse frames
        frame_pattern = re.compile(r"(?:from\s+)?(.+\.rb):(\d+):in\s+`([^\']+)\'")

        for line in lines:
            match = frame_pattern.search(line)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                function_name = match.group(3)

                # Determine frame type
                frame_type = FrameType.APPLICATION
                if "gems" in file_path:
                    frame_type = FrameType.LIBRARY
                elif any(fw in file_path for fw in ["rails", "sinatra", "rack"]):
                    frame_type = FrameType.FRAMEWORK

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    module_name=self._extract_module_from_path(file_path),
                    class_name=None,
                    frame_type=frame_type,
                    language="ruby",
                    raw_frame=line.strip(),
                )
                frames.append(frame)

        return StackTrace(
            trace_id=f"ruby_{datetime.now().timestamp()}",
            language="ruby",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )

    def _extract_module_from_path(self, file_path: str) -> str:
        parts = file_path.split("/")
        if parts[-1].endswith(".rb"):
            return parts[-1][:-3]
        return parts[-1]


class PHPStackTraceParser(StackTraceParser):
    """Parser for PHP stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        php_indicators = [
            ".php:",
            "PHP Fatal error:",
            "PHP Warning:",
            "Stack trace:",
            "#0 /",
            "thrown in",
        ]
        return any(indicator in raw_trace for indicator in php_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "Error"

        # Parse error info
        for line in lines:
            if "PHP Fatal error:" in line or "PHP Warning:" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    error_type = parts[1].strip()
                    error_message = parts[2].strip()

        # Parse frames
        frame_pattern = re.compile(r"#\d+\s+(.+\.php)\((\d+)\):\s+(.+)")

        for line in lines:
            match = frame_pattern.search(line)
            if match:
                file_path = match.group(1)
                line_number = int(match.group(2))
                function_info = match.group(3)

                # Extract function name
                func_match = re.match(r"([\w\\]+(?:::[\w]+)?)\(", function_info)
                function_name = func_match.group(1) if func_match else function_info

                frame = StackFrame(
                    index=len(frames),
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    module_name=self._extract_module_from_path(file_path),
                    class_name=None,
                    frame_type=FrameType.APPLICATION,
                    language="php",
                    raw_frame=line.strip(),
                )
                frames.append(frame)

        return StackTrace(
            trace_id=f"php_{datetime.now().timestamp()}",
            language="php",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )

    def _extract_module_from_path(self, file_path: str) -> str:
        parts = file_path.split("/")
        if parts[-1].endswith(".php"):
            return parts[-1][:-4]
        return parts[-1]


class RustStackTraceParser(StackTraceParser):
    """Parser for Rust stack traces."""

    def identify_language(self, raw_trace: str) -> bool:
        rust_indicators = [
            "thread 'main' panicked",
            "stack backtrace:",
            ".rs:",
            "at src/",
            "note: run with `RUST_BACKTRACE=1`",
        ]
        return any(indicator in raw_trace for indicator in rust_indicators)

    def parse(self, raw_trace: str) -> StackTrace:
        lines = raw_trace.strip().split("\n")
        frames = []
        error_message = ""
        error_type = "panic"

        # Parse panic message
        for line in lines:
            if "panicked at" in line:
                parts = line.split("panicked at", 1)
                if len(parts) > 1:
                    error_message = parts[1].strip().split(",")[0].strip(" '")

        # Parse frames
        frame_pattern = re.compile(r"\d+:\s+(?:0x[\da-f]+\s+-\s+)?(.+?)(?:::(.+?))?$")

        for line in lines:
            if line.strip().startswith(tuple(str(i) + ":" for i in range(100))):
                match = frame_pattern.search(line)
                if match:
                    module_func = match.group(1)
                    location = match.group(2) if match.group(2) else ""

                    # Extract file and line if present
                    file_path = None
                    line_number = None
                    if "at " in location:
                        loc_parts = location.split("at ", 1)
                        if len(loc_parts) > 1:
                            file_info = loc_parts[1]
                            file_match = re.match(r"(.+\.rs):(\d+)", file_info)
                            if file_match:
                                file_path = file_match.group(1)
                                line_number = int(file_match.group(2))

                    frame = StackFrame(
                        index=len(frames),
                        file_path=file_path,
                        line_number=line_number,
                        function_name=module_func,
                        module_name=(
                            module_func.split("::")[0]
                            if "::" in module_func
                            else module_func
                        ),
                        class_name=None,
                        frame_type=FrameType.APPLICATION,
                        language="rust",
                        raw_frame=line.strip(),
                    )
                    frames.append(frame)

        return StackTrace(
            trace_id=f"rust_{datetime.now().timestamp()}",
            language="rust",
            error_message=error_message,
            error_type=error_type,
            frames=frames,
            timestamp=datetime.now(),
        )
