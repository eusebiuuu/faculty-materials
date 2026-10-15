
### Exploitation

control flow graph: dependencies and basic blocks

control flow hijacking

- A pointer is a variable that stores a memory address that can be modified.
- **A code pointer** is a variable storing a memory address that can be modified pointing to code. It is used as an operand for a jump instruction.

1. Have memory overwrite capability
2. Locate code pointer
3. Locate code area (could be a function) that has code that benefits the attacker
4. overwrite code pointer with address from step 3
5. trigger the use of the code pointer
6. benefit

interpretation of input

code reuse
code injection

[Shell Storm](https://shell-storm.org/shellcode/index.html)