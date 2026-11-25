use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Boolean,
    Int64,
    Float64,
    Utf8,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Null(DataType),
    Boolean(bool),
    Int64(i64),
    Float64(f64),
    Utf8(String),
}

impl ScalarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::Null(dt) => *dt,
            ScalarValue::Boolean(_) => DataType::Boolean,
            ScalarValue::Int64(_) => DataType::Int64,
            ScalarValue::Float64(_) => DataType::Float64,
            ScalarValue::Utf8(_) => DataType::Utf8,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, ScalarValue::Null(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl Field {
    pub fn new(name: &str, data_type: DataType, nullable: bool) -> Self {
        Field {
            name: name.to_string(),
            data_type,
            nullable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Schema {
    pub fields: Vec<Field>,
}

pub type SchemaRef = Arc<Schema>;

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }
    pub fn index_of(&self, name: &str) -> Result<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| EngineError::ColumnNotFound(name.to_string()))
    }
}

#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum EngineError {
    #[error("type error: {0}")]
    Type(String),
    #[error("column '{0}' not found")]
    ColumnNotFound(String),
    #[error("plan error: {0}")]
    Plan(String),
    #[error("arithmetic error: {0}")]
    Arithmetic(String),
    #[error("memory budget exceeded: requested {requested} bytes, available {available}")]
    Memory { requested: usize, available: usize },
    #[error("io error: {0}")]
    Io(String),
    #[error("not implemented: {0}")]
    Unimplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, EngineError>;
