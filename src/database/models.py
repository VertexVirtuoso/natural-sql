"""Database models and data structures."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryResult:
    """Represents the result of a SQL query execution."""
    
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    execution_time: float
    query: str
    timestamp: datetime
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if the query was executed successfully."""
        return self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "execution_time": self.execution_time,
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "success": self.is_success
        }


@dataclass
class TableSchema:
    """Represents a database table schema."""
    
    name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary."""
        return {
            "name": self.name,
            "columns": self.columns,
            "primary_keys": self.primary_keys,
            "foreign_keys": self.foreign_keys
        }


@dataclass
class DatabaseSchema:
    """Represents the complete database schema."""
    
    tables: List[TableSchema]
    database_name: str
    
    def get_table(self, table_name: str) -> Optional[TableSchema]:
        """Get a specific table schema by name."""
        for table in self.tables:
            if table.name.lower() == table_name.lower():
                return table
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the schema to a dictionary."""
        return {
            "database_name": self.database_name,
            "tables": [table.to_dict() for table in self.tables]
        }