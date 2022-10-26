from typing import Generator, Optional, Union
from cocos.tokenizecode import CodeParsingOutput, CodeParser, CodeTokenizer, Span, Point


class WrappedParser:
    """
    In some languages code can't be parsed without errors unless it is wrapped in a function or a class.
    This class wraps the code in a function or a class and then parses it. Returns the best result with
    the least amount of errors.
    """

    def __init__(self, parser: Union[CodeParser, CodeTokenizer]):
        self.parser = parser
        self.wrapper: dict[str, list[str]] = {
            "java": [
                "java_inside_class",
                "java_inside_method",
            ],
            "php": [
                "php_inside_class",
            ],
        }

    def parse(
        self, code: str, language: str, name: Optional[str] = None
    ) -> Union[CodeParsingOutput, tuple[CodeParsingOutput, str]]:
        if name is None:
            return self._parse_min_errors(code, language)
        elif name == "none":
            return self.parser.parse(code, language)
        else:
            wrapper = getattr(self, f"wrap_{name}")
            unwrapper = getattr(self, f"unwrap_{name}")
            wrapped_code = wrapper(code)
            parsing_output = self.parser.parse(wrapped_code, language)
            return unwrapper(parsing_output)

    def _parse_min_errors(
        self, code: str, language: str
    ) -> tuple[CodeParsingOutput, str]:
        results = [
            (self.parser.parse(code, language), "none")  # try to parse the code as is
        ]
        parsing_functions = self.wrapper.get(language, [])
        for name in parsing_functions:
            wrapper = getattr(self, f"wrap_{name}")
            unwrapper = getattr(self, f"unwrap_{name}")
            wrapped_code = wrapper(code)
            parsing_output = self.parser.parse(wrapped_code, language)
            results.append((unwrapper(parsing_output), name))
        return min(results, key=lambda x: x[0].num_errors)

    @staticmethod
    def wrap_java_inside_method(code: str) -> str:
        return f"public class Wrapper {{ public void method() {{{code}}}}}"

    @staticmethod
    def unwrap_java_inside_method(
        parsing_output: CodeParsingOutput,
    ) -> CodeParsingOutput:
        if parsing_output.tree.get_node_data(0) == "[ERROR]":
            # then we cant unwrap
            return parsing_output
        tree = parsing_output.tree[26].detach()

        if tree.get_node_data(0) != "[block]":
            # then we cant unwrap
            return parsing_output  # with all its errors

        nodes_to_remove = []
        if tree.get_node_data(1) == "{":
            nodes_to_remove.append(1)
        if tree.get_node_data(len(tree) - 1) == "}":
            nodes_to_remove.append(len(tree) - 1)

        if len(nodes_to_remove) > 1:
            tree = tree.delete_nodes(nodes_to_remove)
        elif len(nodes_to_remove) == 1:
            tree = tree.delete_node(nodes_to_remove[0])
        else:
            print("no nodes to remove")
            parsing_output.tree.pprint()

        parsing_output.tree = tree

        num_bytes_added_before = parsing_output.positions[26].start_byte
        num_bytes_added_before += 1  # for the {

        # todo remove the added positions
        new_positions = []
        for i, position in enumerate(parsing_output.positions[26:-2]):
            if i == 1:
                continue  # the added bracket {

            new_position = Span(
                start_byte=position.start_byte - num_bytes_added_before,
                end_byte=position.end_byte - num_bytes_added_before,
                start_point=Point(
                    row=position.start_point.row,
                    column=(
                        position.start_point.column
                        if position.start_point.row > 0
                        else position.start_point.column - num_bytes_added_before
                    ),
                ),
                end_point=Point(
                    row=position.end_point.row,
                    column=(
                        position.end_point.column
                        if position.end_point.row > 0
                        else position.end_point.column - num_bytes_added_before
                    ),
                ),
            )

            new_positions.append(new_position)

        parsing_output.positions = new_positions

        return parsing_output

    @staticmethod
    def wrap_java_inside_class(code: str) -> str:
        return "public class Wrapper {" + f"{code}" + "\n}"

    @staticmethod
    def unwrap_java_inside_class(
        parsing_output: CodeParsingOutput,
    ) -> CodeParsingOutput:
        if parsing_output.tree.get_node_data(0) == "[ERROR]":
            # then we cant unwrap
            return parsing_output

        tree = parsing_output.tree[10]
        if tree.get_node_data(0) != "[class_body]":
            # then we cant unwrap
            return parsing_output  # with all its errors

        nodes_to_remove = []
        if tree.get_node_data(1) == "{":
            nodes_to_remove.append(1)
        if tree.get_node_data(len(tree) - 1) == "}":
            nodes_to_remove.append(len(tree) - 1)

        if len(nodes_to_remove) > 1:
            tree = tree.delete_nodes(nodes_to_remove)
        elif len(nodes_to_remove) == 1:
            tree = tree.delete_node(nodes_to_remove[0])
        else:
            print("no nodes to remove")
            parsing_output.tree.pprint()

        # remove \n from the end
        tree.node_data[-1] = tree.node_data[-1][:-1]

        parsing_output.tree = tree
        num_bytes_added_before = parsing_output.positions[10].start_byte

        # todo remove the added positions
        new_positions = []
        for i, position in enumerate(parsing_output.positions[10:-1]):
            if i == 1:
                num_bytes_added_before += 1  # for the {
                continue  # the added bracket {

            new_position = Span(
                start_byte=position.start_byte - num_bytes_added_before,
                end_byte=position.end_byte - num_bytes_added_before,
                start_point=Point(
                    row=position.start_point.row,
                    column=(
                        position.start_point.column
                        if position.start_point.row > 0
                        else position.start_point.column - num_bytes_added_before
                    ),
                ),
                end_point=Point(
                    row=position.end_point.row,
                    column=(
                        position.end_point.column
                        if position.end_point.row > 0
                        else position.end_point.column - num_bytes_added_before
                    ),
                ),
            )

            new_positions.append(new_position)

        # remove the last \n
        last_span = new_positions[-1]
        new_positions[-1] = Span(
            start_byte=last_span.start_byte,
            end_byte=last_span.end_byte - 1,
            start_point=last_span.start_point,
            end_point=Point(
                row=last_span.end_point.row - 1,
                column=last_span.end_point.column,
            ),
        )

        parsing_output.positions = new_positions
        return parsing_output

    @staticmethod
    def wrap_php_inside_class(code: str) -> str:
        return "<?php class Wrapper {" + f"{code}" + "} ?>"

    @staticmethod
    def unwrap_php_inside_class(parsing_output: CodeParsingOutput) -> CodeParsingOutput:
        method_tree = parsing_output.tree[12]
        try:
            assert method_tree.get_node_data(0) == "[method_declaration]"
        except AssertionError as e:
            parsing_output.tree.pprint()
            print("#" * 100)
            print("Fatal could not detect method declaration.")
            print(parsing_output.num_errors)
            parsing_output.tree.pprint()
            raise e

        parsing_output.tree = method_tree

        # added only a single row

        return parsing_output
