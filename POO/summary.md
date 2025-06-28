# Table of contents


---

# Summary materials

1. Definitions of objects, classes and OOP
2. Checklist
	- Write the code not looking at the given one and notice problems
	- Triggers: const, static, virtual, const int &
	- Explain briefly the code
	- Private inheritance
3. References, pointers and functions
	   - A reference is an implicit pointer that acts like another name for the referenced object / variable
	   - References must be always initialised because it points always to the same memory address (exceptions: function params, member data in classes, returned value)
	   - One cannot get the address of a reference, create an array of references or cannot reference bytes
	   - The formal params are those created in the function initiation and destroyed in the function termination
	   - Reference call - copy of the param address
	   - Value call - copy of the value in a formal param
	   - Function identification in overloading: number of params, returned type
	   - Default parameters must be specified only once (either in definition, either in implementation)
	   - Arrays of pointers (\*\*p) (generalisation)
	   - Pointer incrementation is done by increasing the address that is pointed by the pointer with the exact data type of the pointer
	   - Pointers to class members
	   - Pointers of class T conversion to pointers of class U
	   - If a class has a reference as data member, then each object of that class has as data member this variable on exact same address; thus each modification made from an object is visible from all the other objects
	   - Classes with references as member data need `=` operator overloaded
	   - Constants and references can be initialised in classes only with the initialisation constructor list (and not the constructor's body), because the class cannot access the private member data of the sub-object (like `const int` or references)
	   - References don't respect upcasting
	   - Temporary objects are const
	   - `this` is a constant pointer to an object
	   - Modifications of the reference represent modifications of the referred value (duh)
	   - When a function returns an object, a temporary object is created to store the value; after the value was returned the object is then destroyed
	   - Default return type is int
	   - `mutable` variables in lambda functions persist throughout the functions calls, but, if not copied by reference, are not modified by them
	   - if in a lambda function the capture (i.e. the content in `[]`) contains `&` individually, then we can add only variables copied by value; in the same manner, if the capture contains `=` individually one can add only variables copied by reference; also it resolves in an error if one wants to add the same variable in capture
	   - When an expression expects a lvalue as member one cannot provide a rvalue instead without errors (lvalues: references, pointers, constants etc., rvalue: objects, concrete values etc.); exception: constant reference

4. Constants
    - One cannot implicitly change the data types of pointers of different occupied space (see double to int example)
    - `Const T* == T const*` and `(const) T* const`
    - Classes with constant data types need `=` operator and CC overloaded
    - `T x; Const T *x = &x` is allowed, but the reverse is not! (`const T x; T *x = (T*)&x` is allowed but considered a bad practice)
    - Constants default data types: 10 - `int`, 10.1 - `double`, 'c' - `char`
    - Data types ambiguity appears when multiple conversions are possible (eg.: int to double or float)
    - `T x = const T a` is valid syntax
    - A non-const reference cannot reference a const object
    - Returning consts by value in functions has no special meaning for built-in types, but with pointers it does
    - When a variable is declared as `const` the `internal linkage` mechanism is triggered i.e. it cannot be access from external files and the program doesn't allocate memory for it (exceptions: access by reference, data type with complex structure, exception: `external const`)
    - Constant methods can only be called from constant objects
    - Static constants must be initialised at the declaration
    - Variables declared with the keyword `mutable` in classes can be modified within const functions

5. Relation between objects
	- Aggregation / composition: composition of an object of more simple (and independent / dependent) objects
    - A class cannot have value members of same class data type, but can have pointers to it
    - Encapsulation represents the separation of internal (and private) functionalities of an object to the external (and public) ones
    - Inline functions in classes: explicit and implicit inlining

6. Class related types
	   - Struct has default as public and can have functions (in c++)
	   - Union are similar to structs, except for the fact that any data type references the same memory location (wholly or divided - in arrays)
	   - The inheritance, copying, references or statics are not defined on union types
	   - No objects with constructors or destructors can be union members
	   - Anonymous unions: if global, must be static & cannot have functions (thus, they don't accept private and protected)

7. Operators
    - Algebraic operators should not modify the operands (like `+`, `*` etc.) and should return a non-referenced object when declared as methods
    - Cannot be overloaded: `.`, `? :`, `::`, `.*`, `sizeof`, `typeid`
    - Cannot be overloaded with friend functions: `=`, `()`, `[]`, `->`
    - We cannot have default parameters (exception: `()`)
    - Friend operator functions usually need referenced parameters
    - `New` and `delete` operators overloading
    - Overloading the `,` operator
    - Overloading the prefix incrementation vs postfix incrementation operators (dummy param and reference)

8. Constructors and destructors
	   - When a class has a pointer as data member the copy constructor (and `=` operator if the case) must be overridden to prevent memory leaks (especially if there are functions called with objects of this class by value, not by reference)
	   - Calling order within single objects: base classes (in the order of the declaration), then there are the local objects (composition), then the current class's constructor
	   - The destructors are in reverse order **always**
	   - Self made copy constructor hides the default one
	   - Initialisation list of constructors makes optional the local class data initialisation
	   - On dynamic memory allocation the CC, destructors and `=` operator must be redefined (**rule of 3**)
	   - If the CC is overridden in the derived class it **has the duty to transfer** the base class member data
	   - Pseudo-constructors allow predefined data types to be treated as objects in the context of initialisation
	   - Objects with one parameter constructors of type T are declared same as the data type T

9. Static
    - Static objects are destroyed only at the end, after all objects from the same scope are destroyed
    - Static data are variables unique for the class they are instantiated in, but same for all objects of that class; they are stored in memory independent of the objects
    - Static functions can access only static data (non-static can access all types of data)
    - Static functions cannot access the alias this
    - Static objects need the predefined constructor
    - Non-const static variables must be initialised after the class declaration
    - Static objects trigger `internal linkage` (unless `external` is added)
    - Static and global objects, even if created after local objects, are destroyed last in the reverse order of creation

10. Nested classes
    - The implementations of defined functions of nested classes are written outside the most external class
    - Nested classes cannot access non-static members of outside classes
    - Classes inside functions cannot have static members, but can have static functions
    - Local classes (inside functions) can access local static variables and global variables
    - Classes inside functions must have the implementation of their functions inline implicit
    - Declaration and definition can be separated (but resolution operator must be used) (the implementation is outside external class)
	- Local classes can access data and functions from other local classes from the same function (siblings)

11. Inheritance
    - `virtual` in classes adds a pointer named **vptr** in class; thus the class allocated memory grows and, also, triggers late binding and polymorphism
    - **The class memory allocation is done considering the alignment with the step being the size of the biggest data member (maximum being 8 bytes)**
    - The functions (if not virtual) don't occupy memory
    - A string has size of 32 bytes
    - Upcasting cannot be done on private inheritance
    - Restrictive inheritance transforms the looser types from the base class into the access modifier type in the derived class (and the narrower ones remain the same)
    - To tackle this, one can use the `using ClassName::attributeName;` syntax
    - **Virtual functions are always overridden when possible (eg.: upcasting) - late binding**
    - If the virtual keyword is not present even if the base class object is instantiated with a derived object (pointer or reference), due to the lack of virtual keyword the function called is from the base class, not the derived class (explanation: vptr, vtable and constructors)
    - When virtual functions are overloaded, the late-binding mechanism cannot take place (we cannot call even the same name functions from the base class when working with derived class objects)
    - Covariance is a method that allows virtual functions across an hierarchy to have different return types, but the return type from a parent should be a base type for a return type of a child
    - Covariance behaves not quite as we used to think: the return type is the one returned from the class type of the object that calls the function (even though it can be casted into others); thus any assignment of a derived class with the result of such function called from a base object (even though it is a derived class) is not supported
    - When a derived class calls a virtual function inside its constructor (inside the base constructor), the function from the base class is called (explanation: vptr points to the base class's vtable)
    - In inheritance, the sizes of derived classes add up with the ones from the base classes with the step as the biggest type from the base classes and the inner ones
    - Virtual prioritizes the class initialisation
    - Functions with the same name but different return type or number of params in the derived class hide the ones with the same name in the base class
    - From a base class pointer one cannot access the methods from a derived class (even if it can downcast to it) unless it's known (like virtual etc.) or casted
    - A static function cannot be virtual
    - For proper multiple non-disjoint inheritance (diamond inheritance) implementation each path from the common base class to the chosen derived class must have at least one virtual
    - Polymorphism: at run time (virtual, downcasting, typeid), at compilation time
    - Dynamic cast iff polymorphic hierarchy
    - In polymorphic hierarchies the destructor is recommended to be virtual
    - A class without members has size of 1 (canonical byte)
    - Dynamic cast works iff the hierarchy is polymorphic (otherwise one can use simple casting `((Derivata&) baseInstanceThatReferencesADerivedObject).f()`, but not assignment `Derivata& b = a;` where `a` has type of base class, is wrong)
    - `Static_cast` can work on non-polymorphic hierarchies and is similar to standard casting (from C)
    - `Const_cast` is used to remove (or add) the `const` or `volatile` qualifiers, but **the types must be the same**
    - `Volatile` is a qualifier that tells the compiler to not optimise the variable due to possible unexpected modifications (by hardware, other threads etc.)
    - `Reinterpret_cast` converts a data type into another fundamentally different (usually pointer to pointer / int / unsigned type; otherwise it is most probably undefined behaviour)
    - Runtime polymorphism: moștenire, funcții virtuale, upcasting
    - Compilation polymorphism: functions overloading, operators overloading, templates
    - `Typeinfo` ChatGPT
    - `Typeid(obj).name()` gets the typename of the `obj`; it's an example of RTTI, thus one can determine the real type of an object using it similar to `dynamic_cast`; it's independent of `const` and on null values it throws `std::bad_typeid` error; on references it yields the type of the referred object, not the reference itself; it works as expected only on polymorphic hierarchies

12. Exceptions
    - try, catch, throw - nothing special
    - terminate is called when the throw is not caught
    - to catch an error one must use an exact (or looser) data type
    - functions can specify what data types can be thrown: `ret_type funct_name(U1, ..., Un) throw(T1, ..., Tn) {}`
    - `noexcept` is equivalent to `throw()`

13. Templates
    - the order of function templates is the following:
        - if the function has `<>`: check for template specialisation (`<>`), then check the template (`<T>`) (1 params, 2 params...), then remake the process for looser types (with non-template functions)
        - otherwise (the function has no brackets): check for the exact function (without templates), then check for template specialisation, then check for the template (1 params, 2 params...), then remake the process for looser types (with non-template functions)
    - values of basic types (like `int` or `char` etc.) are implicitly `const` but can be transformed into non-const without problems
    - in template classes, if a class function is implemented outside the class, it must be declared as template; this is because all functions in a template class are by default templates too

14. STL
    - `push_back()`:  in vectors it reallocates spaces, calling the copy constructor for each element within once a new element is added; in lists it reallocates without copying the previous elements
    - `auto` deducts the type from the right operand; `decltype` guess it
    - `reverse_iterator`
    - `rbegin` and `rend` iterators -> + to go reverse
    - `back` returns a reference to the last element
    - `erase` - value (erase all occurrences) & iter (erase all elements from the iterator's position (start -> end))
    - `list` container represents a double linked list
    - `container<T, allocator<T>>` - the container is represented in memory with the given allocator