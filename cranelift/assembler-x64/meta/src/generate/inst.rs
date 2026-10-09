use super::{Formatter, fmtln, generate_derive, generate_derive_arbitrary_bounds};
use crate::dsl;

impl dsl::Inst {
    /// `struct <inst> { <op>: Reg, <op>: Reg, ... }`
    pub fn generate_struct(&self, f: &mut Formatter) {
        let struct_name = self.struct_name_with_generic();
        let where_clause = if self.requires_generic() {
            "where R: Registers"
        } else {
            ""
        };

        fmtln!(f, "/// `{self}`");
        generate_derive(f);
        if self.requires_generic() {
            generate_derive_arbitrary_bounds(f);
        }
        f.add_block(&format!("pub struct {struct_name} {where_clause}"), |f| {
            for k in &self.format.operands {
                let loc = k.location;
                let ty = k.generate_type();
                fmtln!(f, "pub {loc}: {ty},");
            }

            if self.has_trap {
                fmtln!(f, "pub trap: TrapCode,");
            }
        });
    }

    fn requires_generic(&self) -> bool {
        self.format.uses_register()
    }

    /// `<struct_name><R>`
    pub(crate) fn struct_name_with_generic(&self) -> String {
        let struct_name = self.name();
        if self.requires_generic() {
            format!("{struct_name}<R>")
        } else {
            struct_name
        }
    }

    /// `impl...`
    fn generate_impl_block_start(&self) -> &str {
        if self.requires_generic() {
            "impl<R: Registers>"
        } else {
            "impl"
        }
    }

    /// `impl <inst> { ... }`
    pub fn generate_struct_impl(&self, f: &mut Formatter) {
        let impl_block = self.generate_impl_block_start();
        let struct_name = self.struct_name_with_generic();
        f.add_block(&format!("{impl_block} {struct_name}"), |f| {
            self.generate_memory_access_function(f);
            f.empty_line();
            self.generate_explicit_memory_function(f);
            f.empty_line();
            self.generate_stack_access_function(f);
            f.empty_line();
            self.generate_written_gprs_function(f);
            f.empty_line();
            self.generate_new_function(f);
            f.empty_line();
            self.generate_mnemonic_function(f);
            f.empty_line();
            self.generate_encode_function(f);
            f.empty_line();
            self.generate_visit_function(f);
            f.empty_line();
            self.generate_is_available_function(f);
            f.empty_line();
            self.generate_features_function(f);
            f.empty_line();
            self.generate_num_registers_function(f);
        });
    }

    fn generate_explicit_memory_function(&self, f: &mut Formatter) {
        use dsl::OperandKind::*;
        let generic = if self.requires_generic() {
            ""
        } else {
            "<R: Registers>"
        };
        f.add_block(&format!("pub fn explicit_memory{generic}(&self) -> Option<crate::MemoryOperand<'_, R::ReadGpr>>"), |f| {
            if matches!(self.mnemonic.as_str(), "leaw" | "leal" | "leaq" | "nop" | "nopl") {
                fmtln!(f, "None");
                return;
            }
            let operands = self.format.operands.iter().filter(|o| o.location.uses_memory()).collect::<Vec<_>>();
            assert!(operands.len() <= 1, "multiple explicit memory operands need an ordered audit");
            let Some(operand) = operands.first() else {
                fmtln!(f, "None");
                return;
            };
            let loc = operand.location;
            let access = match (operand.mutability.is_read(), operand.mutability.is_write()) {
                (true, false) => "Read",
                (false, true) => "Write",
                (true, true) => "ReadWrite",
                (false, false) => unreachable!("memory operand without an effect"),
            };
            // Lane instructions use rm32 in the DSL to match disassembly, but
            // access one/two bytes in their memory form.
            let bytes = match self.mnemonic.as_str() {
                "pextrb" | "vpextrb" | "pinsrb" | "vpinsrb" => 1,
                "pextrw" | "vpextrw" | "pinsrw" | "vpinsrw" => 2,
                _ => loc.bytes(),
            };
            let value = format!("Some(crate::MemoryOperand {{ address, access: crate::MemoryAccess::{access}, bytes: {bytes} }})");
            match loc.kind() {
                Mem(_) => {
                    fmtln!(f, "let address = &self.{loc};");
                    fmtln!(f, "{value}");
                }
                RegMem(_) => f.add_block(&format!("match &self.{loc}"), |f| {
                    let class = loc.reg_class().unwrap();
                    fmtln!(f, "{class}Mem::Mem(address) => {value},");
                    fmtln!(f, "{class}Mem::{class}(..) => None,");
                }),
                _ => unreachable!(),
            }
        });
    }

    fn generate_stack_access_function(&self, f: &mut Formatter) {
        f.add_block("pub fn stack_access(&self) -> crate::StackAccess", |f| {
            let effect = match self.mnemonic.as_str() {
                "pushw" => "Push { bytes: 2 }",
                "pushq" => "Push { bytes: 8 }",
                "popw" => "Pop { bytes: 2 }",
                "popq" => "Pop { bytes: 8 }",
                "retq" => "Return",
                "callq" => "Call",
                _ => "None",
            };
            fmtln!(f, "crate::StackAccess::{effect}");
        });
    }

    fn generate_written_gprs_function(&self, f: &mut Formatter) {
        use dsl::OperandKind::*;
        f.add_block("pub fn written_gprs(&self) -> alloc::vec::Vec<u8>", |f| {
            let mut writes = Vec::new();
            for op in &self.format.operands {
                if !op.mutability.is_write()
                    || !matches!(op.location.reg_class(), Some(dsl::RegClass::Gpr))
                {
                    continue;
                }
                let loc = op.location;
                match loc.kind() {
                    FixedReg(_) => writes.push(format!("self.{loc}.0.enc()")),
                    Reg(_) => writes.push(format!("self.{loc}.as_ref().enc()")),
                    RegMem(_) => {
                        fmtln!(f, "let mut writes = alloc::vec::Vec::new();");
                        f.add_block(&format!("if let GprMem::Gpr(reg) = &self.{loc}"), |f| {
                            fmtln!(f, "writes.push(reg.enc());");
                        });
                        for write in &writes {
                            fmtln!(f, "writes.push({write});");
                        }
                        // There can be further implicit fixed writes (CMPXCHG).
                        let position = self
                            .format
                            .operands
                            .iter()
                            .position(|x| x.location == loc)
                            .unwrap();
                        for other in self.format.operands.iter().skip(position + 1) {
                            if other.mutability.is_write()
                                && matches!(other.location.kind(), FixedReg(_))
                                && matches!(other.location.reg_class(), Some(dsl::RegClass::Gpr))
                            {
                                let other = other.location;
                                fmtln!(f, "writes.push(self.{other}.0.enc());");
                            }
                        }
                        if matches!(self.mnemonic.as_str(), "popw" | "popq") {
                            fmtln!(f, "writes.push(crate::gpr::enc::RSP);");
                        }
                        fmtln!(f, "writes");
                        return;
                    }
                    _ => unreachable!(),
                }
            }
            if matches!(
                self.mnemonic.as_str(),
                "pushw" | "pushq" | "popw" | "popq" | "retq" | "callq"
            ) {
                writes.push("crate::gpr::enc::RSP".into());
            }
            fmtln!(f, "alloc::vec![{}]", writes.join(", "));
        });
    }

    /// Generate memory effects from operand mutability, with the complete set
    /// of implicit-memory/address-only exceptions in the current instruction DSL.
    fn generate_memory_access_function(&self, f: &mut Formatter) {
        use dsl::OperandKind::*;
        f.add_block("pub fn memory_access(&self) -> crate::MemoryAccess", |f| {
            let special = match self.mnemonic.as_str() {
                "leaw" | "leal" | "leaq" | "nop" | "nopl" => Some("None"),
                "callq" => Some("Call"),
                "retq" => Some("Read"),
                "lfence" => Some("Fence"),
                _ => None,
            };
            if let Some(special) = special {
                fmtln!(f, "crate::MemoryAccess::{special}");
                return;
            }
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            match self.mnemonic.as_str() {
                "pushw" | "pushq" => writes.push("true".to_string()),
                "popw" | "popq" => reads.push("true".to_string()),
                _ => {}
            }
            for operand in &self.format.operands {
                let loc = operand.location;
                let condition = match loc.kind() {
                    Mem(_) => "true".to_string(),
                    RegMem(_) => {
                        let class = loc.reg_class().unwrap();
                        format!("matches!(&self.{loc}, {class}Mem::Mem(_))")
                    }
                    FixedReg(_) | Reg(_) | Imm(_) => continue,
                };
                if operand.mutability.is_read() {
                    reads.push(condition.clone());
                }
                if operand.mutability.is_write() {
                    writes.push(condition);
                }
            }
            let reads = if reads.is_empty() {
                "false".to_string()
            } else {
                reads.join(" || ")
            };
            let writes = if writes.is_empty() {
                "false".to_string()
            } else {
                writes.join(" || ")
            };
            f.add_block(&format!("match ({reads}, {writes})"), |f| {
                fmtln!(f, "(false, false) => crate::MemoryAccess::None,");
                fmtln!(f, "(true, false) => crate::MemoryAccess::Read,");
                fmtln!(f, "(false, true) => crate::MemoryAccess::Write,");
                fmtln!(f, "(true, true) => crate::MemoryAccess::ReadWrite,");
            });
        });
    }

    // `fn new(<params>) -> Self { ... }`
    pub fn generate_new_function(&self, f: &mut Formatter) {
        let params = comma_join(
            self.format
                .operands
                .iter()
                .map(|o| format!("{}: impl Into<{}>", o.location, o.generate_type()))
                .chain(if self.has_trap {
                    Some("trap: impl Into<TrapCode>".to_string())
                } else {
                    None
                }),
        );
        fmtln!(f, "#[must_use]");
        f.add_block(&format!("pub fn new({params}) -> Self"), |f| {
            f.add_block("Self", |f| {
                for o in &self.format.operands {
                    let loc = o.location;
                    fmtln!(f, "{loc}: {loc}.into(),");
                }
                if self.has_trap {
                    fmtln!(f, "trap: trap.into(),");
                }
            });
        });
    }

    /// `fn mnemonic(&self) -> &'static str { ... }`
    pub fn generate_mnemonic_function(&self, f: &mut Formatter) {
        use dsl::Customization::*;

        fmtln!(f, "#[must_use]");
        fmtln!(f, "#[inline]");
        f.add_block(
            &format!("pub fn mnemonic(&self) -> alloc::borrow::Cow<'static, str>"),
            |f| {
                if self.custom.contains(Mnemonic) {
                    fmtln!(f, "crate::custom::mnemonic::{}(self)", self.name());
                } else {
                    fmtln!(f, "alloc::borrow::Cow::Borrowed(\"{}\")", self.mnemonic);
                }
            },
        );
    }

    /// `fn encode(&self, ...) { ... }`
    fn generate_encode_function(&self, f: &mut Formatter) {
        use dsl::Customization::*;

        f.add_block(
            &format!("pub fn encode(&self, buf: &mut impl CodeSink)"),
            |f| {
                if self.custom.contains(Encode) {
                    fmtln!(f, "crate::custom::encode::{}(self, buf);", self.name());
                } else {
                    self.generate_possible_trap(f);
                    match &self.encoding {
                        dsl::Encoding::Rex(rex) => self.format.generate_rex_encoding(f, rex),
                        dsl::Encoding::Vex(vex) => self.format.generate_vex_encoding(f, vex),
                        dsl::Encoding::Evex(evex) => self.format.generate_evex_encoding(f, evex),
                    }
                }
            },
        );
    }

    // `buf.add_trap(...)`
    fn generate_possible_trap(&self, f: &mut Formatter) {
        if self.has_trap {
            f.comment("Emit trap.");
            fmtln!(f, "buf.add_trap(self.trap);");
        } else if let Some(op) = self.format.uses_memory() {
            use dsl::OperandKind::*;
            f.comment("Emit trap.");
            match op.kind() {
                Mem(_) => {
                    f.add_block(
                        &format!("if let Some(trap_code) = self.{op}.trap_code()"),
                        |f| {
                            fmtln!(f, "buf.add_trap(trap_code);");
                        },
                    );
                }
                RegMem(_) => {
                    let ty = op.reg_class().unwrap();
                    f.add_block(&format!("if let {ty}Mem::Mem({op}) = &self.{op}"), |f| {
                        f.add_block(&format!("if let Some(trap_code) = {op}.trap_code()"), |f| {
                            fmtln!(f, "buf.add_trap(trap_code);");
                        });
                    });
                }
                _ => unreachable!(),
            }
        }
    }

    /// `fn visit(&self, ...) { ... }`
    fn generate_visit_function(&self, f: &mut Formatter) {
        use dsl::{Customization::*, OperandKind::*};
        let extra_generic_bound = if self.requires_generic() {
            ""
        } else {
            "<R: Registers>"
        };
        let visitor = if self.format.operands.is_empty() && !self.custom.contains(Visit) {
            "_"
        } else {
            "visitor"
        };
        f.add_block(&format!("pub fn visit{extra_generic_bound}(&mut self, {visitor}: &mut impl RegisterVisitor<R>)"), |f| {
            if self.custom.contains(Visit) {
                fmtln!(f, "crate::custom::visit::{}(self, visitor)", self.name());
                return;
            }
            for o in &self.format.operands {
                let mutability = o.mutability.generate_snake_case();
                let reg = o.location.reg_class();
                match o.location.kind() {
                    Imm(_) => {
                        // Immediates do not need register allocation.
                        //
                        // If an instruction happens to only have immediates
                        // then generate a dummy use of the `visitor` variable
                        // to suppress unused variables warnings.
                        fmtln!(f, "let _ = visitor;");
                    }
                    FixedReg(loc) => {
                        let reg_lower = reg.unwrap().to_string().to_lowercase();
                        fmtln!(f, "let enc = self.{loc}.expected_enc();");
                        fmtln!(f, "visitor.fixed_{mutability}_{reg_lower}(&mut self.{loc}.0, enc);");
                    }
                    Reg(loc) => {
                        let reg_lower = reg.unwrap().to_string().to_lowercase();
                        fmtln!(f, "visitor.{mutability}_{reg_lower}(self.{loc}.as_mut());");
                    }
                    RegMem(loc) => {
                        let reg = reg.unwrap();
                        let reg_lower = reg.to_string().to_lowercase();
                        fmtln!(f, "visitor.{mutability}_{reg_lower}_mem(&mut self.{loc});");
                    }
                    Mem(loc) => {
                        // Note that this is always "read" because from a
                        // regalloc perspective when using an amode it means
                        // that the while a write is happening that's to
                        // memory, not registers.
                        fmtln!(f, "visitor.read_amode(&mut self.{loc});");
                    }

                }
            }
        });
    }

    /// `fn is_available(&self, ...) -> bool { ... }`
    fn generate_is_available_function(&self, f: &mut Formatter) {
        fmtln!(f, "#[must_use]");
        f.add_block(
            "pub fn is_available(&self, features: &impl AvailableFeatures) -> bool",
            |f| {
                let expr = self.features.generate_boolean_expr("features");
                fmtln!(f, "{expr}");
            },
        );
    }

    /// `fn features(&self) -> Features { ... }`
    fn generate_features_function(&self, f: &mut Formatter) {
        fmtln!(f, "#[must_use]");
        f.add_block("pub fn features(&self) -> &'static Features", |f| {
            self.features.generate_constructor_expr(f);
        });
    }

    /// `fn num_registers_available(&self) -> usize { ... }`
    fn generate_num_registers_function(&self, f: &mut Formatter) {
        fmtln!(f, "#[must_use]");
        f.add_block("pub fn num_registers_available(&self) -> usize", |f| {
            use dsl::Encoding::*;
            let n = match self.encoding {
                Rex(_) | Vex(_) => 16,
                Evex(_) => 32,
            };
            fmtln!(f, "{n}")
        });
    }

    /// `impl Display for <inst> { ... }`
    pub fn generate_display_impl(&self, f: &mut Formatter) {
        use crate::dsl::Customization::*;
        let impl_block = self.generate_impl_block_start();
        let struct_name = self.struct_name_with_generic();
        f.add_block(
            &format!("{impl_block} core::fmt::Display for {struct_name}"),
            |f| {
                f.add_block(
                    "fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result",
                    |f| {
                        if self.custom.contains(Display) {
                            fmtln!(f, "crate::custom::display::{}(f, self)", self.name());
                            return;
                        }

                        fmtln!(f, "let name = self.mnemonic();");
                        if self.format.operands.is_empty() {
                            fmtln!(f, "f.write_str(&name)");
                            return;
                        }
                        for op in self.format.operands.iter() {
                            let location = op.location;
                            let to_string = location.generate_to_string(op.extension);
                            fmtln!(f, "let {location} = {to_string};");
                        }
                        let ordered_ops = self.format.generate_att_style_operands();
                        let mut implicit_ops = self.format.generate_implicit_operands();
                        if self.has_trap {
                            fmtln!(f, "let trap = self.trap;");
                            if implicit_ops.is_empty() {
                                implicit_ops.push_str(" ;; {trap}");
                            } else {
                                implicit_ops.push_str(", {trap}");
                            }
                        }
                        fmtln!(f, "write!(f, \"{{name}} {ordered_ops}{implicit_ops}\")");
                    },
                );
            },
        );
    }

    /// `impl From<struct> for Inst { ... }`
    pub fn generate_from_impl(&self, f: &mut Formatter) {
        let struct_name_r = self.struct_name_with_generic();
        let variant_name = self.name();
        f.add_block(
            &format!("impl<R: Registers> From<{struct_name_r}> for Inst<R>"),
            |f| {
                f.add_block(&format!("fn from(inst: {struct_name_r}) -> Self"), |f| {
                    fmtln!(f, "Self::{variant_name}(inst)");
                });
            },
        );
    }
}

fn comma_join<S: Into<String>>(items: impl Iterator<Item = S>) -> String {
    items.map(Into::into).collect::<Vec<_>>().join(", ")
}
