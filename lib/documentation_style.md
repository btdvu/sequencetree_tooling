# Documentation Style Guide

This document defines the documentation and commenting style for the sequencetree_tooling codebase.

## Module-Level Docstrings

Every Python module must begin with a comprehensive docstring that includes:

1. **Title**: Brief, descriptive title on first line
2. **Description**: 1-3 sentence overview of module functionality
3. **Author**: `Author: Brian-Tinh Vu`
4. **Date**: Creation/modification date in format `MM/DD/YYYY`
5. **Dependencies**: List of required packages (e.g., `numpy, sigpy, scipy`)
6. **Citations** (if applicable): Academic references with full citation details including:
   - Author names
   - Paper title
   - Conference/journal information
   - DOI when available
7. **Quick Start**: 2-4 example usage lines showing common function calls

### Module Docstring Template

```python
"""
[Module Title]

[1-3 sentence description of module purpose and capabilities.]

Author: Brian-Tinh Vu
Date: MM/DD/YYYY
Dependencies: [package1, package2, ...]

Citations:
    [If applicable, list academic references with full details]

Quick Start:
    [example_call_1]
    [example_call_2]
"""
```

## Naming Conventions

### Function Names

**REQUIRED:** All function names must use **strict camelCase** format.

- **First word**: lowercase
- **Subsequent words**: capitalize first letter only
- **NO underscores** in function names
- **NO module name prefixes** in function names

**Correct Examples:**
- `trajGoldenAngle` (not `traj_golden_angle` or `TrajGoldenAngle`)
- `estGradDelay` (not `est_grad_delay` or `EstGradDelay`)
- `readoutLine` (not `readout_line` or `ReadoutLine`)

**Incorrect Examples:**
- ❌ `read_twix_siemens_XA` - uses underscores instead of camelCase
- ❌ `read_twix_from_cartesian_mask` - uses underscores instead of camelCase
- ❌ `trajGoldenAngleStackOfStars` in `stack_of_stars.py` - includes module name "StackOfStars"
- ❌ `estGradDelayRampSampling` in `ramp_sampling.py` - includes module name "RampSampling"

**Correct Versions:**
- ✅ `readSiemensXA` or `readXA` (in `twix_io.py`)
- ✅ `readFromCartesianMask` or `readCartesian` (in `twix_io.py`)
- ✅ `trajGoldenAngle` (in `stack_of_stars.py` - module context makes it clear it's stack-of-stars)
- ✅ `estGradDelay` (in `ramp_sampling.py` - module context makes it clear it's ramp sampling)

### Module/File Names

**REQUIRED:** Module filenames must use **snake_case** (underscores as word separators).

**Examples:**
- `ramp_sampling.py`
- `stack_of_stars.py`
- `twix_io.py`
- `radial.py`

### Function Arguments and Variables

**REQUIRED:** All function arguments and local variables must use **snake_case**.

**Examples:**
- `n_spokes_per_slice`
- `N_plat`
- `shape_i`
- `starting_angle`

### Redundancy Rule

**CRITICAL:** Function names must NOT include the module name to avoid redundancy.

The module provides context through the import statement. Users call functions as `module.function()`, so repeating the module name in the function name is redundant.

**Bad Practice:**
```python
# In twix_io.py
def read_twix_siemens_XA(datfile, nviews):  # ❌ "twix" is redundant
    pass

# Usage: twix_io.read_twix_siemens_XA()  # "twix" appears twice!
```

**Good Practice:**
```python
# In twix_io.py
def readSiemensXA(datfile, nviews):  # ✅ No redundancy
    pass

# Usage: twix_io.readSiemensXA()  # Clear and concise
```

### Enforcement Checklist

Before committing code, verify:

1. ✅ All function names use strict camelCase (no underscores)
2. ✅ Function names do NOT contain the module name
3. ✅ Module filenames use snake_case
4. ✅ All function arguments use snake_case
5. ✅ Function names start with lowercase letter

## Function Docstrings

All public functions must use **NumPy-style docstrings** with the following sections:

### Required Sections

1. **Brief Description**: One-line summary immediately after opening quotes
2. **Extended Description** (optional): Additional context, 1-2 sentences
3. **Parameters**: All function arguments with type and description
4. **Returns**: Return value(s) with type and description
5. **Notes** (optional): Implementation details, algorithms, or usage guidance
6. **Raises** (optional): Exceptions that may be raised
7. **Examples** (optional): Usage examples with expected output

### Function Docstring Format

```python
def function_name(param1, param2, optional_param=default):
    """
    Brief one-line description of function purpose.
    
    [Optional: Extended description providing additional context
    about the function's behavior or implementation.]
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    optional_param : type, optional
        Description of optional parameter. Default: value.
    
    Returns
    -------
    return_name : type
        Description of return value.
        Additional details about shape, format, or contents.
    
    Raises
    ------
    ExceptionType
        When this exception occurs.
    
    Notes
    -----
    - Implementation details
    - Algorithm references
    - Usage guidance
    
    Examples
    --------
    >>> result = function_name(arg1, arg2)
    >>> print(result.shape)
    (128, 128)
    """
```

### Parameter Documentation Style

- **Type specification**: Use descriptive types (`int`, `np.ndarray`, `str`, `bool`)
- **Array shapes**: Include shape information for NumPy arrays (e.g., `shape: (n_spokes, N, 2)`)
- **Optional parameters**: Mark with `, optional` and specify default value
- **Enums**: Use `{'option1', 'option2'}` format for limited choices

### Return Documentation Style

- **Named returns**: Give return values descriptive names
- **Shape details**: For arrays, specify dimensions and what each represents
- **Multiple returns**: Document each return value separately

## Inline Comments

### When to Use Inline Comments

1. **Section headers**: Mark major code sections with descriptive comments
2. **Algorithm steps**: Explain non-obvious computational steps
3. **Variable initialization**: Clarify purpose of key variables
4. **Complex operations**: Explain mathematical or logical operations

### Inline Comment Style

- Place comments **above** the code block they describe
- Use complete sentences with proper capitalization
- Keep comments concise but informative
- Avoid redundant comments that simply restate the code

### Examples

```python
# Initialize trajectory array
traj = np.zeros((n_spokes, N, 2), dtype=float)

# Golden-angle increment (degrees)
PI = 3.141592
angle_inc = 111.25

# Generate each radial spoke
for i_projection in range(n_spokes):
    # Calculate spoke angle
    angle_in_deg = i_projection * angle_inc + starting_angle
```

## Private Function Documentation

Private functions (prefixed with `_`) should have minimal but clear docstrings:

- Brief one-line description
- Parameters section (concise)
- Returns section (concise)
- Notes section only if algorithm is non-trivial

Private functions typically omit Examples and extended descriptions.

## Special Conventions

### TODO Comments

Use standard TODO format for incomplete features:
```python
# TODO: add a safety check to see if there are any additional ADCs that were acquired that we did not read in
```

### Progress Indicators

For long-running operations, include user-friendly progress output:
```python
print(f"Reading: {datfile}\nProgress...")
# ... processing loop ...
print(f"{percentFinished:3d} %")
```

### Error Messages

Provide descriptive error messages with context:
```python
raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")
raise EOFError("Unexpected end of file while reading readout count.")
```

## Formatting Standards

- **Line length**: Keep docstrings readable; wrap at reasonable length
- **Blank lines**: One blank line after docstring, before first code
- **Indentation**: Match function indentation for docstring content
- **Punctuation**: End descriptions with periods

## Documentation Completeness

### Must Document
- All public functions and classes
- All module files
- Complex algorithms or mathematical operations
- File format specifications
- Coordinate system conventions

### May Omit Documentation
- Trivial getters/setters
- Self-explanatory variable assignments
- Standard Python idioms

## Cross-References

When referencing related functions or methods:
- Mention function names in Notes sections
- Reference academic papers for algorithms
- Link to related functionality in module docstring

## Mathematical Notation

- Use clear variable names in code that match mathematical notation
- Explain coordinate systems and conventions in Notes
- Reference equations from papers when implementing published methods
