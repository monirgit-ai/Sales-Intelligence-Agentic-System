"""
Base Agent Class
Abstract base class for all agents in the system.
Provides common functionality and interface for agentic operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the Sales Intelligence Agentic System.
    
    This class provides a foundation for agentic intelligence and will be
    extended to work with LangChain for advanced agent capabilities.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
            max_iterations: Maximum number of iterations for agent loops
            verbose: Whether to log detailed information
        """
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.iteration_count = 0
        self.history: List[Dict[str, Any]] = []
        
        if self.verbose:
            logger.info(f"Initialized agent: {self.name}")
    
    def reset(self):
        """Reset agent state."""
        self.iteration_count = 0
        self.history.clear()
    
    def log_action(self, action: str, result: Any, metadata: Optional[Dict] = None):
        """
        Log an action taken by the agent.
        
        Args:
            action: Description of the action
            result: Result of the action
            metadata: Additional metadata about the action
        """
        log_entry = {
            "iteration": self.iteration_count,
            "action": action,
            "result": result,
            "metadata": metadata or {}
        }
        self.history.append(log_entry)
        
        if self.verbose:
            logger.info(f"[{self.name}] {action}: {result}")
    
    def check_iteration_limit(self) -> bool:
        """
        Check if iteration limit has been reached.
        
        Returns:
            True if under limit, False if limit reached
        """
        if self.iteration_count >= self.max_iterations:
            logger.warning(
                f"[{self.name}] Maximum iterations ({self.max_iterations}) reached"
            )
            return False
        return True
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iteration_count += 1
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        This method should be implemented by subclasses.
        
        Args:
            input_data: Dictionary containing input data and context
            
        Returns:
            Dictionary containing results and insights
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this agent provides.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "history_length": len(self.history),
            "capabilities": self.get_capabilities()
        }

