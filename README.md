# SOFTWARE TESTING PROJECT
------------------------------------------
By - Abhinav Chawla(IMT2013002)
   - Udbhav Vats(IMT2013055)
------------------------------------------
# AIM - MUTATION SOURCE CODE
A project based on mutation operators applied at the level of a statement within a method or a function. The mutated program needs to be strongly killed by the designed test cases. At least three different mutation operators should be used.
------------------------------------------
# THEORY-
Mutation Testing - Mutation testing (or Mutation analysis or Program mutation) evaluates the quality of software tests. It is a type of White Box Testing which is mainly used for Unit Testing. Mutation testing involves modifying a program’s source code or byte code in small ways. Following are the steps to execute mutation testing:

Step 1: Faults are introduced into the source code of the program by creating many versions called mutants. Each mutant should contain a single fault, and the goal is to cause the mutant version to fail which demonstrates the effectiveness of the test cases.

Step 2: Test cases are applied to the original program and also to the mutant program. A Test Case should be adequate, and it is tweaked to detect faults in a program.

Step 3: Compare the results of original and mutant program.

Step 4: If the original program and mutant programs generate different output, then that the mutant is killed by the test case. Hence the test case is good enough to detect the change between the original and the mutant program.

Step 5: If the original program and mutant program generate the same output, Mutant is kept alive. In such cases, more effective test cases need to be created that kill all mutants.

# MUTATION SCORE = (Killed Mutants / Total number of Mutants) * 100

------------------------------------------
# LANGUAGE USED - Python 3.5.1
------------------------------------------
# TOOLS-
MutPy - MutPy is a mutation testing tool for Python 3.x source code. MutPy supports standard unittest module, generates YAML reports and has colorful output. It’s apply mutation on AST level. You could boost your mutation testing process with high order mutations (HOM) and code coverage analysis.
-------------------------------------------
# WHAT IS OUR PROJECT ABOUT
We have used a code which contains a structure of tree and contains various functions called on tree, as specified in the documentation. Also, we are using the source code which compared various sorting algorithms, specified in documentation. We have written multiple test cases for all the functions in the tests.py file while the source code is the source.py file.
-------------------------------------------
# CONTRIBUTION-
Udbhav Vats - Source Code and Test cases for Sorting Algorithms. Documentation of the source code.
Abhinav Chawla - Source Code and Test cases for  Tree based algorithms. Written the README.md and Documentation of source code partial.
-------------------------------------------
# HOW TO RUN -
1. First install mutpy using the follwoing command in terminal: $pip3 install mutpy
2. Then type the following command in terminal: $mut.py --target source --unit-test tests -m
    where source.py is the source file name while tests.py is the file containing the various test cases
--------------------------------------------
# RESULTS-
Mutation score [18.87361 s]: 91.5%
   - all: 282
   - killed: 247 (87.6%)
   - survived: 23 (8.2%)
   - incompetent: 12 (4.3%)
   - timeout: 0 (0.0%)

------------------------------------------
# MUTATION OPERATORS IN MUTPY
AOD - arithmetic operator deletion
AOR - arithmetic operator replacement
ASR - assignment operator replacement
BCR - break continue replacement
COD - conditional operator deletion
COI - conditional operator insertion
CRP - constant replacement
DDL - decorator deletion
EHD - exception handler deletion
EXS - exception swallowing
IHD - hiding variable deletion
IOD - overriding method deletion
IOP - overridden method calling position change
LCR - logical connector replacement
LOD - logical operator deletion
LOR - logical operator replacement
ROR - relational operator replacement
SCD - super calling deletion
SCI - super calling insert
SIR - slice index remove
--------------------------------------------
