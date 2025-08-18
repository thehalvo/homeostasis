# ADR-007: Monitoring Data Retention Policy

Technical Story: #MON-001

## Context

The Homeostasis framework generates significant amounts of monitoring data including logs, metrics, error traces, and patch history. Without a clear retention policy, storage costs will grow unbounded, queries will become slower, and compliance requirements may be violated. We need a data retention strategy that balances operational needs, compliance requirements, and cost efficiency.

## Decision Drivers

- Storage cost management
- Query performance requirements
- Compliance and audit requirements
- Debugging and analysis needs
- Privacy regulations (GDPR, CCPA)
- Disaster recovery requirements
- Machine learning training data needs

## Considered Options

1. **Infinite Retention** - Keep all data forever
2. **Fixed Time-Based** - Delete all data after X days
3. **Tiered Retention** - Different retention for different data types
4. **Sampling-Based** - Keep samples of old data
5. **Value-Based Retention** - Keep data based on importance

## Decision Outcome

Chosen option: "Tiered Retention", implementing different retention periods for different types of data based on their operational value and compliance requirements, because it optimizes storage costs while meeting all operational and regulatory needs.

### Positive Consequences

- **Cost Optimization**: Expensive hot storage for recent data only
- **Performance**: Fast queries on recent, relevant data
- **Compliance**: Meets various regulatory requirements
- **Flexibility**: Different policies for different data types
- **ML Training**: Preserves valuable training data
- **Debugging**: Recent issues fully debuggable
- **Scalability**: Predictable storage growth

### Negative Consequences

- **Complexity**: Multiple retention rules to manage
- **Data Loss**: Old data eventually deleted
- **Migration Overhead**: Moving data between tiers
- **Query Complexity**: Different queries for different tiers
- **Recovery Limitations**: Can't debug very old issues
- **Storage Management**: Multiple storage systems

## Implementation Details

### Data Tiers

```python
class DataTier(Enum):
    HOT = "hot"        # 0-7 days - Full detail, instant access
    WARM = "warm"      # 8-30 days - Full detail, slower access
    COLD = "cold"      # 31-90 days - Compressed, slow access
    ARCHIVE = "archive" # 91-365 days - Highly compressed, rare access
    PURGE = "purge"    # >365 days - Deleted (except compliance data)
```

### Retention Policies by Data Type

```yaml
retention_policies:
  error_logs:
    hot_days: 7
    warm_days: 23  # Total 30 days
    cold_days: 60  # Total 90 days
    archive_days: 275  # Total 365 days
    purge_after: 365
    
  metrics:
    hot_days: 7
    warm_days: 23
    cold_days: 60
    archive_days: 90  # Total 180 days
    purge_after: 180
    aggregation:
      - interval: hourly
        retain_days: 30
      - interval: daily
        retain_days: 365
      - interval: monthly
        retain_days: 1825  # 5 years
  
  patch_history:
    hot_days: 30
    warm_days: 60
    cold_days: 275
    archive_days: 1460  # Total 5 years
    purge_after: 1825  # Keep 5 years for audit
    
  security_events:
    hot_days: 90
    warm_days: 275
    cold_days: 1825  # Total 7 years
    archive_days: 0
    purge_after: 2555  # 7 years for compliance
    
  user_activity:
    hot_days: 30
    warm_days: 60
    cold_days: 0
    archive_days: 0
    purge_after: 90  # GDPR compliance
```

### Storage Backend Mapping

```python
TIER_STORAGE_MAPPING = {
    DataTier.HOT: {
        'backend': 'elasticsearch',
        'replication': 3,
        'ssd': True,
        'compression': None
    },
    DataTier.WARM: {
        'backend': 'elasticsearch',
        'replication': 2,
        'ssd': False,
        'compression': 'best_speed'
    },
    DataTier.COLD: {
        'backend': 's3',
        'storage_class': 'STANDARD_IA',
        'compression': 'gzip'
    },
    DataTier.ARCHIVE: {
        'backend': 's3',
        'storage_class': 'GLACIER',
        'compression': 'best_compression'
    }
}
```

### Data Migration Pipeline

```python
class DataMigrationPipeline:
    async def migrate_data(self):
        for data_type, policy in self.policies.items():
            await self._migrate_to_warm(data_type, policy)
            await self._migrate_to_cold(data_type, policy)
            await self._migrate_to_archive(data_type, policy)
            await self._purge_expired(data_type, policy)
    
    async def _migrate_to_warm(self, data_type: str, policy: dict):
        cutoff_date = datetime.now() - timedelta(days=policy['hot_days'])
        
        data = await self.hot_storage.query(
            data_type=data_type,
            date_range={'lt': cutoff_date}
        )
        
        if data:
            await self.warm_storage.bulk_insert(data)
            await self.hot_storage.delete(data_ids=[d.id for d in data])
            
            self.metrics.record('data_migration', {
                'from': 'hot',
                'to': 'warm',
                'count': len(data),
                'data_type': data_type
            })
```

### Aggregation Strategy

```python
class DataAggregator:
    def aggregate_metrics(self, metrics: List[Metric], interval: str) -> List[AggregatedMetric]:
        aggregations = {
            'hourly': self._aggregate_hourly,
            'daily': self._aggregate_daily,
            'monthly': self._aggregate_monthly
        }
        
        return aggregations[interval](metrics)
    
    def _aggregate_hourly(self, metrics: List[Metric]) -> List[AggregatedMetric]:
        # Group by hour and calculate statistics
        hourly_groups = defaultdict(list)
        
        for metric in metrics:
            hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_groups[hour].append(metric.value)
        
        return [
            AggregatedMetric(
                timestamp=hour,
                min=min(values),
                max=max(values),
                avg=statistics.mean(values),
                p50=statistics.median(values),
                p95=numpy.percentile(values, 95),
                p99=numpy.percentile(values, 99),
                count=len(values)
            )
            for hour, values in hourly_groups.items()
        ]
```

### Query Interface

```python
class TieredQueryInterface:
    async def query(self, query: Query) -> QueryResult:
        # Determine which tiers to query based on date range
        tiers = self._get_relevant_tiers(query.date_range)
        
        # Query each tier in parallel
        results = await asyncio.gather(*[
            self._query_tier(tier, query) for tier in tiers
        ])
        
        # Merge and sort results
        merged = self._merge_results(results)
        
        return QueryResult(
            data=merged,
            tiers_queried=tiers,
            performance_stats=self._get_query_stats()
        )
    
    def _get_relevant_tiers(self, date_range: DateRange) -> List[DataTier]:
        tiers = []
        now = datetime.now()
        
        if date_range.end >= now - timedelta(days=7):
            tiers.append(DataTier.HOT)
        if date_range.end >= now - timedelta(days=30):
            tiers.append(DataTier.WARM)
        if date_range.end >= now - timedelta(days=90):
            tiers.append(DataTier.COLD)
        if date_range.start < now - timedelta(days=90):
            tiers.append(DataTier.ARCHIVE)
            
        return tiers
```

### Compliance and Audit

```python
class ComplianceManager:
    def __init__(self):
        self.regulations = {
            'gdpr': GDPRCompliance(),
            'ccpa': CCPACompliance(),
            'hipaa': HIPAACompliance(),
            'sox': SOXCompliance()
        }
    
    def validate_retention_policy(self, policy: RetentionPolicy) -> ValidationResult:
        violations = []
        
        for regulation_name, regulation in self.regulations.items():
            if regulation.applies_to(policy.data_type):
                result = regulation.validate(policy)
                if not result.valid:
                    violations.extend(result.violations)
        
        return ValidationResult(
            valid=len(violations) == 0,
            violations=violations
        )
```

### Backup and Recovery

```python
class BackupStrategy:
    def get_backup_policy(self, tier: DataTier) -> BackupPolicy:
        policies = {
            DataTier.HOT: BackupPolicy(
                frequency='continuous',
                retention_days=7,
                geo_redundant=True
            ),
            DataTier.WARM: BackupPolicy(
                frequency='daily',
                retention_days=30,
                geo_redundant=True
            ),
            DataTier.COLD: BackupPolicy(
                frequency='weekly',
                retention_days=90,
                geo_redundant=False
            ),
            DataTier.ARCHIVE: BackupPolicy(
                frequency='monthly',
                retention_days=365,
                geo_redundant=False
            )
        }
        return policies[tier]
```

## Links

- [Monitoring Module Documentation](../../modules/monitoring/README.md)
- [ADR-010: Performance Monitoring Approach](010-performance-monitoring-approach.md)
- [GDPR Compliance Guide](../regulated_industries_guide.md)