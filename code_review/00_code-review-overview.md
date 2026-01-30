# AI-Assisted Code Review System for MRI Pulse Sequence and Image Reconstruction Development

## Overview

This code review system provides a structured, multi-mode approach to improve your Python code for MRI pulse sequence programming and medical image reconstruction. Each mode focuses on specific aspects of code quality, enabling you to learn from targeted feedback and continuously improve your craft.

## Philosophy

Code reviews are learning opportunities. Each review should:
- **Identify specific problems** with clear explanations of why they matter
- **Propose concrete solutions** with code examples when applicable
- **Teach domain-specific best practices** for MRI physics, k-space, and sequences
- **Build robust, maintainable code** that meets medical device quality standards

## How to Use This System

### 1. Complete Your First Working Draft
Write your module/function until it produces correct results. Don't worry about perfection—focus on getting it working first.

### 2. Choose Your Review Mode(s)
Select one or more review modes based on your current priorities:

- **Code Style Mode**: Enforce your established naming conventions, documentation standards, and formatting rules
- **Correctness Mode**: Check for logic bugs, numerical issues, array handling, and algorithmic errors
- **Performance Mode**: Identify bottlenecks when processing large 3D/4D volumes
- **Test Coverage Mode**: Find missing edge cases, failure modes, and validation gaps
- **Safety & Compliance Mode**: Verify SAR limits, HIPAA compliance, and FDA requirements
- **API Design Mode**: Improve usability, naming, type hints, and documentation
- **Maintainability Mode**: Reduce technical debt, eliminate duplication, simplify complexity
- **MRI Domain Mode**: Validate MRI physics, k-space operations, coil handling, and sequence logic

### 3. Submit to AI Reviewer
Provide your code along with the selected review mode prompt to Claude (Opus or Sonnet) or Windsurf AI.

### 4. Review Feedback and Learn
- Read each issue carefully and understand the underlying problem
- Implement suggested fixes
- Note patterns to avoid in future code
- Ask clarifying questions if needed

### 5. Iterate as Needed
Run multiple review modes to get comprehensive feedback. Start with Correctness, then Style, then others based on your needs.

## Review Mode Summary

| Mode | Primary Focus | When to Use |
|------|---------------|-------------|
| **Code Style** | Naming, docs, formatting | Every commit before merge |
| **Correctness** | Logic bugs, numerical errors | After initial working draft |
| **Performance** | Speed, memory, scalability | When processing large datasets |
| **Test Coverage** | Edge cases, failure modes | Before production deployment |
| **Safety & Compliance** | SAR, HIPAA, FDA | Medical device sequences |
| **API Design** | Usability, interface clarity | Public-facing functions |
| **Maintainability** | Tech debt, complexity | Regular refactoring cycles |
| **MRI Domain** | Physics, k-space, sequences | All MRI-specific code |

## Best Practices

### For Code Authors
1. **Self-review first**: Read through your code before submitting for review
2. **Provide context**: Explain what the code does and any design decisions
3. **Be specific**: If you have concerns about specific sections, call them out
4. **Stay open**: View feedback as learning, not criticism
5. **Test first**: Ensure basic functionality works before review

### For Using AI Reviewers
1. **One mode at a time**: Don't overwhelm yourself with all modes simultaneously
2. **Prioritize**: Start with Correctness and Code Style, then branch out
3. **Provide test data**: Include example inputs and expected outputs when relevant
4. **Include documentation**: Share related docstrings or design notes
5. **Iterate**: Fix critical issues, then re-review if needed

## Expected Outcomes

After using this system regularly, you should see:

✓ **Fewer bugs** through systematic correctness checks  
✓ **Faster code** through performance optimization  
✓ **Better documentation** through style enforcement  
✓ **More robust code** through comprehensive test coverage  
✓ **Safer sequences** through compliance verification  
✓ **Cleaner APIs** through usability feedback  
✓ **Maintainable codebase** through complexity management  
✓ **Domain expertise** through MRI-specific guidance  

## Integration with Your Workflow

This system complements your established code style (documented separately) and provides the structured review process you've been missing as a solo developer. Think of it as having an experienced MRI software engineer reviewing your code and teaching you best practices.

## Getting Started

1. Read through each review mode file to understand what it covers
2. Select the most relevant mode for your current code
3. Copy the mode prompt into your conversation with Claude or Windsurf
4. Paste your code and any relevant context
5. Review the feedback and implement improvements
6. Repeat for other modes as needed

Happy coding! 🧲🔬
