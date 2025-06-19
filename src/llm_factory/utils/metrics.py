"""
Metrics collection utilities.
"""

import time
from typing import Dict, List, Optional

from loguru import logger


class MetricsCollector:
    """Collect and track metrics for LLM usage."""
    
    def __init__(self):
        self.metrics: List[Dict] = []
    
    def record_request(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Optional[float] = None,
        latency: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record a request metric."""
        metric = {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            "latency": latency,
            "success": success,
            "error": error,
        }
        
        self.metrics.append(metric)
        logger.info(f"Recorded metric: {metric}")
    
    def get_summary(self, provider: Optional[str] = None) -> Dict:
        """Get summary statistics."""
        filtered_metrics = self.metrics
        if provider:
            filtered_metrics = [m for m in self.metrics if m["provider"] == provider]
        
        if not filtered_metrics:
            return {"total_requests": 0}
        
        total_requests = len(filtered_metrics)
        successful_requests = len([m for m in filtered_metrics if m["success"]])
        total_tokens = sum(m["total_tokens"] for m in filtered_metrics)
        total_cost = sum(m["cost"] for m in filtered_metrics if m["cost"])
        
        latencies = [m["latency"] for m in filtered_metrics if m["latency"]]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_latency": avg_latency,
            "provider": provider,
        }
    
    def get_provider_breakdown(self) -> Dict[str, Dict]:
        """Get metrics breakdown by provider."""
        providers = set(m["provider"] for m in self.metrics)
        return {provider: self.get_summary(provider) for provider in providers}
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        logger.info("Cleared all metrics")
