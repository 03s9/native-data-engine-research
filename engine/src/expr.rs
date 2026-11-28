//! Typed expression AST (spec §2.2). FROZEN: variants/signatures are the slice contract.
use crate::types::{DataType, ScalarValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    Neg,
    IsNull,
    IsNotNull,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column {
        name: String,
        index: usize,
        data_type: DataType,
    },
    Literal(ScalarValue),
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
        return_type: DataType,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
        return_type: DataType,
    },
    Cast {
        expr: Box<Expr>,
        target_type: DataType,
    },
    InList {
        expr: Box<Expr>,
        list: Vec<Expr>,
        negated: bool,
    },
    Case {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        return_type: DataType,
    },
}

impl Expr {
    pub fn return_type(&self) -> DataType {
        match self {
            Expr::Column { data_type, .. } => *data_type,
            Expr::Literal(v) => v.data_type(),
            Expr::Binary { return_type, .. }
            | Expr::Unary { return_type, .. }
            | Expr::Case { return_type, .. } => *return_type,
            Expr::Cast { target_type, .. } => *target_type,
            Expr::InList { .. } => DataType::Boolean,
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Expr::Column { .. } | Expr::Literal(_) => 1,
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + expr.depth(),
            Expr::Binary { left, right, .. } => 1 + left.depth().max(right.depth()),
            Expr::InList { expr, list, .. } => {
                1 + expr
                    .depth()
                    .max(list.iter().map(|e| e.depth()).max().unwrap_or(0))
            }
            Expr::Case {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                1 + condition
                    .depth()
                    .max(then_expr.depth())
                    .max(else_expr.depth())
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Expr::Column { .. } | Expr::Literal(_) => 1,
            Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => 1 + expr.node_count(),
            Expr::Binary { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Expr::InList { expr, list, .. } => {
                1 + expr.node_count() + list.iter().map(|e| e.node_count()).sum::<usize>()
            }
            Expr::Case {
                condition,
                then_expr,
                else_expr,
                ..
            } => 1 + condition.node_count() + then_expr.node_count() + else_expr.node_count(),
        }
    }
}
