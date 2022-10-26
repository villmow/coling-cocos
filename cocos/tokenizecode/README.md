# tokenizecode

`tokenizecode` provides an easy interface to [tree_sitter](https://tree-sitter.github.io/tree-sitter/) for machine learning on source code. It parses source code into syntax trees, where every leaf node corresponds to actual code in the source file.

## Features

- Parses source code into concrete syntax trees (similar to ASTs, but they contain every token including whitespace).
- Implements Byte Pair Encoding (BPE) for tree leaves.

## Usage

### Parsing

Here is an example of how to parse Java code using `tokenizecode`:

```python
from cocos.tokenizecode import CodeParser, CodeParsingOutput

java_code = """class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}"""

# Create a parser object with the default grammars
parser = CodeParser()

# Parse the code and return a syntax tree
out: CodeParsingOutput = parser.parse(java_code, "java")

print(f"Found {out.num_errors} errors while parsing the code.")
print("The syntax tree is:")
out.tree.pprint()
```
The example above will output:

```
Found 0 errors while parsing the code.
The syntax tree is:
TensorTree():
  0. [program]
  ╰── 1. [class_declaration]
      ├── 2. class
      ├── 3. ·
      ├── 4. [identifier]
      │   ╰── 5. HelloWorld
      ├── 6. ·
      ╰── 7. [class_body]
          ├── 8. {
          ├── 9. ⏎····
          ├── 10. [method_declaration]
          │   ├── 11. [modifiers]
          │   │   ├── 12. public
          │   │   ├── 13. ·
          │   │   ╰── 14. static
          │   ├── 15. ·
          │   ├── 16. [void_type]
          │   │   ╰── 17. void
          │   ├── 18. ·
          │   ├── 19. [identifier]
          │   │   ╰── 20. main
          │   ├── 21. [formal_parameters]
          │   │   ├── 22. (
          │   │   ├── 23. [formal_parameter]
          │   │   │   ├── 24. [array_type]
          │   │   │   │   ├── 25. [type_identifier]
          │   │   │   │   │   ╰── 26. String
          │   │   │   │   ╰── 27. [dimensions]
          │   │   │   │       ├── 28. [
          │   │   │   │       ╰── 29. ]
          │   │   │   ├── 30. ·
          │   │   │   ╰── 31. [identifier]
          │   │   │       ╰── 32. args
          │   │   ╰── 33. )
          │   ├── 34. ·
          │   ╰── 35. [block]
          │       ├── 36. {
          │       ├── 37. ⏎········
          │       ├── 38. [expression_statement]
          │       │   ├── 39. [method_invocation]
          │       │   │   ├── 40. [field_access]
          │       │   │   │   ├── 41. [identifier]
          │       │   │   │   │   ╰── 42. System
          │       │   │   │   ├── 43. .
          │       │   │   │   ╰── 44. [identifier]
          │       │   │   │       ╰── 45. out
          │       │   │   ├── 46. .
          │       │   │   ├── 47. [identifier]
          │       │   │   │   ╰── 48. println
          │       │   │   ╰── 49. [argument_list]
          │       │   │       ├── 50. (
          │       │   │       ├── 51. [string_literal]
          │       │   │       │   ╰── 52. "Hello,·World!"
          │       │   │       ╰── 53. )
          │       │   ╰── 54. ;
          │       ├── 55. ·⏎····
          │       ╰── 56. }
          ├── 57. ⏎
          ╰── 58. }
```

### Tokenization and BPE Encoding

After parsing, you can binarize the tree including BPE encoding:

```python
from cocos.tokenizecode import CodeTokenizer

# the CodeTokenizer combines a Parser and a BPE tokenizer
tokenizer = CodeTokenizer()

tree = tokenizer.encode_to_tree(java_code, "java")
# tree = tokenizer.encode_to_tree(out) or this

tree.pprint()
```

The `CodeTokenizer` will output a tree of integers:

```
TensorTree():
  0. 31140
  ╰── 1. 31094
      ├── 2. 357
      ├── 3. 14
      ├── 4. 31383
      │   ╰── 5. 5
      │       ├── 6. 10236
      │       ╰── 7. 8692
      ├── 8. 14
      ╰── 9. 31016
          ├── 10. 105
          ╰── ...
```

### Decoding

You can decode the tree of integers back to a tree of strings:

```python
decoded_tree = tokenizer.decode_tree(tree, keep_bpe=True)
decoded_tree.pprint()
```
```
TensorTree():
  0. [program]
  ╰── 1. [class_declaration]
      ├── 2. class
      ├── 3. ·
      ├── 4. [identifier]
      │   ╰── 5. [BPE]
      │       ├── 6. Hello
      │       ╰── 7. World
      ├── 8. ·
      ╰── 9. [class_body]
          ├── 10. {
          ╰── ...
```

In this example, the `HelloWorld` leaf node has been split into `Hello` and `World`, with a new `BPE` node created.
