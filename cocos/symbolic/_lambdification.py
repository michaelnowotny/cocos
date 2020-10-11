import collections
import numbers
import typing as tp

import numpy as np
import sympy as sym

from cocos import numerics as cn
from cocos.numerics.data_types import NumericArray, NumericArrayOrScalar
from cocos.utilities import check_and_make_sequence
from cocos.numerics.numerical_package_selector import (
    select_num_pack,
    get_gpu_and_num_pack_by_dtype_from_iterable
)

from cocos.symbolic.translations import COCOS_TRANSLATIONS
from cocos.symbolic.utilities import find_length_of_state_vectors


DUMMY_TIME_SYMBOL = sym.Symbol('t')


################################################################################
# Works for both Matrices and Arrays
################################################################################
def _compute_replacement_functions(
        state_vectors: tp.List[NumericArrayOrScalar],
        number_of_state_variables: int,
        t: tp.Optional[float] = None,
        replacement_functions: tp.Optional[tp.Tuple[tp.Callable, ...]] = None) \
        -> tp.Tuple[tp.List, int]:
    """
    If an expression has been optimized with common subexpression elimination,
    one first needs to compute the replacement values (which are additional
    arguments to the original function) using so-called replacement functions.
    This is done in an incremental fashion, i.e. the replacement value resulting
    from a given replacement function may be used as an argument of the next
    replacement function.

    This function evaluates these replacement functions with the original
    arguments and returns the replacement values appended to the original
    arguments along with the number of replications.

    Args:
        replacement_functions:
            A sequence of functions that compute the replacement values from the
            original arguments.

        state_vectors: a list of numerical vectors (NumPy or Cocos)
        t: time parameter
        gpu: whether or not to evaluate the function on the gpu
        number_of_state_variables:
            the number of variables (original arguments) in the function

    Returns:
        a tuple of arguments with replacement values appended and the number of
        replications

    """

    # R = max([state_argument.size
    #          for state_argument
    #          in state_vectors
    #          if isinstance(state_argument, (np.ndarray, cn.ndarray))] + [1])

    R = find_length_of_state_vectors(state_vectors)

    if number_of_state_variables != len(state_vectors):
        raise ValueError(f"The number of state vectors({len(state_vectors)}) "
                         f"does not match the number of variables in the "
                         f"function({number_of_state_variables}).")

    arguments = [t] + state_vectors
    if replacement_functions is not None:
        for replacement_function in replacement_functions:
            # Note that the arguments vector grows with each iteration. That
            # means computed common subexpressions from previous iterations are
            # used to evaluate subsequent expressions.

            arguments.append(replacement_function(*arguments))

    return arguments, R


################################################################################
# General Arrays
################################################################################
def lambdify_array_with_modules(
        symbols: tp.Tuple[sym.Symbol, ...],
        array_expression: tp.Union[sym.Array, sym.MatrixBase],
        symbolic_time_function_name_to_numeric_time_function_map:
        tp.Dict[str, tp.Callable],
        modules: tp.Tuple[str, ...]) \
        -> tp.Tuple[tp.Callable, ...]:
    """
    This function takes a SymPy array, unravals the index in Fortran order and
    lambdifies each element separately.

    Args:
        symbols: a tuple of SymPy symbols that are arguments to the function
        array_expression: a SymPy matrix or array
        symbolic_time_function_name_to_numeric_time_function_map: 
            a dictionary mapping the names of functions of time to Python 
            functions
        modules: modules that should be included

    Returns: a tuple of functions of the specified argument representing the
             elements of the symbolic array in Fortran order

    """

    if not isinstance(modules, list):
        modules = list(modules)

    if not (isinstance(array_expression, sym.Array) or
            isinstance(array_expression, sym.MatrixBase)):
        raise TypeError(f"symbolic_matrix_expression must be an instance of "
                        f"sympy.ArrayBase or sympy.MatrixBase "
                        f"but is of type {type(array_expression)}")

    n = (np.prod(array_expression.shape)).item()
    result = []
    for i in range(n):
        index = tuple(np.unravel_index(i, array_expression.shape, order='F'))
        result.append(sym.lambdify(args=symbols,
                                   expr=array_expression[index],
                                   modules
                                   =[symbolic_time_function_name_to_numeric_time_function_map] +
                                     modules))

    return tuple(result)


def lambdify_array(
        symbols: tp.Tuple[sym.Symbol, ...],
        array_expression: tp.Union[sym.Array, sym.MatrixBase],
        symbolic_time_function_name_to_numeric_time_function_map:
        tp.Optional[tp.Dict[str, tp.Callable]] = None) \
        -> tp.Tuple[tp.Tuple[tp.Callable, ...], tp.Tuple[tp.Callable, ...]]:

    if symbolic_time_function_name_to_numeric_time_function_map is None:
        symbolic_time_function_name_to_numeric_time_function_map = dict()

    functions_cpu = \
        lambdify_array_with_modules(
            symbols=symbols,
            array_expression=array_expression,
            symbolic_time_function_name_to_numeric_time_function_map
            =symbolic_time_function_name_to_numeric_time_function_map,
            modules=('numpy', ))

    functions_gpu = \
        lambdify_array_with_modules(
            symbols=symbols,
            array_expression=array_expression,
            symbolic_time_function_name_to_numeric_time_function_map
            =symbolic_time_function_name_to_numeric_time_function_map,
            modules=(COCOS_TRANSLATIONS, ))

    return functions_cpu, functions_gpu


def _compute_result_internal(R: int,
                             dimensions: tp.Tuple[int, ...],
                             arguments: tp.List[NumericArrayOrScalar],
                             functions_cpu: tp.Tuple[tp.Callable, ...],
                             functions_gpu: tp.Tuple[tp.Callable, ...],
                             pre_attach: bool,
                             gpu: bool,
                             dtype: np.generic) \
        -> NumericArray:
    """
    This function evaluates a sequence of functions with given positional
    arguments. Each argument is either a scalar or R-dimensional. The result
    Args:
        R: the number of replications (this is the length each of the input
           vectors if they are not scalars)

        dimensions:
            The shape of the output (excluding the R replications which are
            either attached to the beginning (if pre_attach is True) or to the
            end.

        arguments:
            a list of arguments to be passed to the functions to be evaluated

        functions_cpu: The sequence of functions to be evaluated on the CPU.
        functions_gpu: The sequence of functions to be evaluated on the GPU.
        pre_attach:
            whether to put the replications in the first or the last axis
        gpu: whether to evaluate on the cpu or the gpu
        dtype: the dtype of the output array

    Returns:

    """
    num_pack = select_num_pack(gpu)
    if gpu:
        functions = functions_gpu
    else:
        functions = functions_cpu

    n = (np.prod(dimensions)).item()

    if pre_attach:
        dimension_index = tuple([R] + list(dimensions))
    else:
        dimension_index = tuple(list(dimensions) + [R])

    result = num_pack.zeros(dimension_index, dtype=dtype)

    for i in range(n):
        index = tuple(np.unravel_index(i, dimensions, order='F'))

        function = functions[i]

        if pre_attach:
            location_index = tuple([slice(None)] + list(index))
        else:
            location_index = tuple(list(index) + [slice(None)])

        evaluated_function = function(*arguments)
        result[location_index] = evaluated_function

    return result


################################################################################
# Sequences of Lambdified Array, Matrix, and Vector-Expressions
################################################################################
class LambdifiedArrayExpressions:
    def __init__(
            self,
            symbolic_array_expressions:
            tp.Tuple[tp.Union[sym.Array, sym.MatrixBase], ...],
            argument_symbols: tp.Tuple[sym.Symbol],
            time_symbol: tp.Optional[sym.Symbol] = None,
            symbolic_time_function_name_to_numeric_time_function_map:
            tp.Optional[tp.Dict[str, tp.Callable]] = None,
            squeeze_column_vectors: tp.Optional[tp.Tuple[bool, ...]] = None,
            perform_cse: bool = True,
            lazy_initialization: bool = False,
            pre_attach: tp.Optional[tp.Tuple[bool, ...]] = None,
            dtype: np.generic = np.float32):

        if time_symbol is None:
            if symbolic_time_function_name_to_numeric_time_function_map is not None and \
                    len(symbolic_time_function_name_to_numeric_time_function_map) != 0:
                raise ValueError(
                        'Argument time_symbol must not be none if '
                        'symbolic_time_function_name_to_numeric_time_function_map '
                        'is provided.')

            if any([DUMMY_TIME_SYMBOL
                    in symbolic_array_expression.free_symbols
                    for symbolic_array_expression
                    in symbolic_array_expressions]):
                raise ValueError('Argument time_symbol must be provided '
                                 'explicity if expression contains '
                                 'DUMMY_TIME_SYMBOL.')

            time_symbol = DUMMY_TIME_SYMBOL
            self._time_symbol_provided = False
        else:
            self._time_symbol_provided = True

        if symbolic_time_function_name_to_numeric_time_function_map is None:
            symbolic_time_function_name_to_numeric_time_function_map = dict()

        symbolic_array_expressions \
            = tuple(check_and_make_sequence(symbolic_array_expressions,
                                            sym.MatrixBase))

        # print(f'{symbolic_array_expressions=}\n')
        symbolic_array_expressions = \
                tuple(sym.ImmutableMatrix(symbolic_array_expression)
                      if isinstance(symbolic_array_expression, sym.MatrixBase)
                      else sym.ImmutableDenseNDimArray(symbolic_array_expression)
                      for symbolic_array_expression
                      in symbolic_array_expressions)

        if pre_attach is None:
            pre_attach = len(symbolic_array_expressions) * [True]
        pre_attach = check_and_make_sequence(pre_attach, bool)

        if squeeze_column_vectors is None:
            squeeze_column_vectors = len(symbolic_array_expressions) * [False]
        squeeze_column_vectors = check_and_make_sequence(squeeze_column_vectors,
                                                         bool)

        if not (len(symbolic_array_expressions) == len(pre_attach) and
                len(symbolic_array_expressions) == len(squeeze_column_vectors)):
            raise ValueError("symbolic_matrix_expressions, pre_attach, and "
                             "squeeze_column_vectors must have the same number "
                             "of elements")

        self._argument_symbols = argument_symbols
        self._time_symbol = time_symbol
        self._symbolic_array_expressions = symbolic_array_expressions
        self._symbolic_time_function_name_to_numeric_time_function_map = \
            symbolic_time_function_name_to_numeric_time_function_map
        self._squeeze_column_vectors = squeeze_column_vectors
        self._symbols = \
            tuple([time_symbol] +
                   list(argument_symbols))

        self._perform_cse = perform_cse
        self._pre_attach = pre_attach
        self._dtype = dtype

        self._shapes \
            = tuple([symbolic_array_expression.shape
                     for symbolic_array_expression
                     in symbolic_array_expressions])

        if lazy_initialization:
            self._functions_cpu = None
            self._functions_gpu = None
        else:
            self._perform_initialization()

    def _perform_initialization(self):
        self._functions_cpu = []
        self._functions_gpu = []
        if self._perform_cse:
            repl, redu = sym.cse(self._symbolic_array_expressions,
                                 optimizations='basic')

            replacement_functions_cpu = []
            replacement_functions_gpu = []
            syms = [self._time_symbol] + list(self._argument_symbols)
            for i, v in enumerate(repl):
                replacement_functions_cpu.append(
                    sym.lambdify(args=syms,
                                 expr=v[1],
                                 modules
                                 =[self._symbolic_time_function_name_to_numeric_time_function_map,
                                   'numpy']))

                replacement_functions_gpu.append(
                    sym.lambdify(
                        args=syms,
                        expr=v[1],
                        modules
                        =[self._symbolic_time_function_name_to_numeric_time_function_map,
                          COCOS_TRANSLATIONS]))

                syms.append(v[0])

            self._replacement_functions_cpu = tuple(replacement_functions_cpu)
            self._replacement_functions_gpu = tuple(replacement_functions_gpu)

            for symbolic_matrix_expression in redu:
                functions_cpu, functions_gpu \
                    = lambdify_array(
                        symbols=tuple(syms),
                        array_expression=symbolic_matrix_expression,
                        symbolic_time_function_name_to_numeric_time_function_map
                        =self._symbolic_time_function_name_to_numeric_time_function_map)

                self._functions_cpu.append(functions_cpu)
                self._functions_gpu.append(functions_gpu)
        else:
            for symbolic_matrix_expression in self._symbolic_array_expressions:
                functions_cpu, functions_gpu \
                    = lambdify_array(
                        symbols=self._symbols,
                        array_expression=symbolic_matrix_expression,
                        symbolic_time_function_name_to_numeric_time_function_map
                        =self._symbolic_time_function_name_to_numeric_time_function_map)

                self._functions_cpu.append(functions_cpu)
                self._functions_gpu.append(functions_gpu)

    @property
    def shapes(self) -> tp.Tuple[tp.Tuple[int, ...], ...]:
        return self._shapes

    def is_column_vector(self, i: int) -> bool:
        shape = self._symbolic_array_expressions[i].shape
        return len(shape) == 1 \
               or all([dim == 1
                       for dim
                       in shape[1:]])

    def is_row_vector(self, i: int) -> bool:
        shape = self._symbolic_array_expressions[i].shape
        if len(shape < 2):
            return False
        else:
            return shape[0] == 1 and \
                   (len(shape) == 2 or
                    all([dim == 1
                         for dim
                         in shape[2:]]))

    @property
    def argument_symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._argument_symbols

    @property
    def number_of_state_variables(self) -> int:
        return len(self._argument_symbols)

    @property
    def time_symbol(self) -> sym.Symbol:
        return self._time_symbol

    @property
    def symbolic_array_expressions(self) -> tp.Tuple[sym.ImmutableMatrix, ...]:
        return self._symbolic_array_expressions

    @property
    def symbolic_time_function_name_to_numeric_time_function_map(self) \
            -> tp.Dict[str, tp.Callable]:
        return self._symbolic_time_function_name_to_numeric_time_function_map

    @property
    def symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._symbols

    @property
    def number_of_variables(self) -> int:
        return len(self._symbols)

    @property
    def dtype(self) -> np.generic:
        return self._dtype

    def evaluate_with_list_of_state_vectors(
            self,
            list_of_state_vectors: tp.List[NumericArrayOrScalar],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) \
            -> tp.Tuple[NumericArray, ...]:
        """
        This function evaluates the array expressions at arguments that are
        vectors (arrays with a single axis) or scalars and returns the resulting
        arrays as a tuple.

        Args:
            list_of_state_vectors:
                A list of one-dimensional numeric arrays. The length of this
                list must match the number of arguments to the array-valued
                functions.

            t: time parameter
            gpu: whether to evaluate the array expressions on the GPU

        Returns: a tuple of arrays corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        if t is None:
            if self._time_symbol_provided:
                raise ValueError("Argument 't' must be present if 'time_symbol'"
                                 " was explicitly provided during "
                                 "construction.'")

            if (self._symbolic_time_function_name_to_numeric_time_function_map is not None and
                    len(self._symbolic_time_function_name_to_numeric_time_function_map) > 0):
                raise ValueError(
                        "Argument 't' must be provided if "
                        "symbolic_time_function_name_to_numeric_time_function_map "
                        "is provided.")

        if self._functions_cpu is None or self._functions_gpu is None:
            self._perform_initialization()

        if gpu is None:
            # determine whether to run on cpu or gpu based on the input types
            gpu, _ \
                = get_gpu_and_num_pack_by_dtype_from_iterable(
                list_of_state_vectors)

        if gpu:
            replacement_functions = self._replacement_functions_gpu
        else:
            replacement_functions = self._replacement_functions_cpu

        arguments, R \
            = _compute_replacement_functions(
                state_vectors=list_of_state_vectors,
                number_of_state_variables=self.number_of_state_variables,
                t=t,
                replacement_functions=replacement_functions)

        results = []
        for i, (shape,
                pre_attach,
                functions_cpu,
                functions_gpu,
                squeeze_column_vector) in \
                enumerate(zip(self.shapes,
                              self._pre_attach,
                              self._functions_cpu,
                              self._functions_gpu,
                              self._squeeze_column_vectors)):

            is_column_vector = self.is_column_vector(i)

            if is_column_vector and squeeze_column_vector:
                dimensions = (shape[0], )
            else:
                dimensions = shape

            result = _compute_result_internal(R=R,
                                              dimensions=dimensions,
                                              arguments=arguments,
                                              functions_cpu=functions_cpu,
                                              functions_gpu=functions_gpu,
                                              pre_attach=pre_attach,
                                              gpu=gpu,
                                              dtype=self.dtype)
            results.append(result)

        return tuple(results)

    def evaluate_with_dictionary(
            self,
            symbolic_to_numeric_parameter_map: tp.Dict[sym.Symbol,
                                                       NumericArrayOrScalar],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) \
            -> tp.Tuple[NumericArray, ...]:
        """
        This function evaluates the array expressions at arguments that are
        dictionaries mapping symbolic parameters to one-dimensional numeric
        arrays or scalars and returns the resulting arrays as a tuple.

        Args:
            symbolic_to_numeric_parameter_map:
                A dictionary mapping symbolic parameters to numeric arrays or
                scalars.

            t: time parameter
            gpu: whether to evaluate the array expressions on the GPU

        Returns: a tuple of arrays corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        list_of_state_vectors = \
            [symbolic_to_numeric_parameter_map[symbol]
             for symbol
             in self.argument_symbols]

        return self.evaluate_with_list_of_state_vectors(
                        list_of_state_vectors=list_of_state_vectors,
                        t=t,
                        gpu=gpu)

    def evaluate_with_kwargs(self,
                             t: tp.Optional[float] = None,
                             gpu: tp.Optional[bool] = None,
                             **kwargs) \
            -> tp.Tuple[NumericArray, ...]:
        """
        This function evaluates the array expressions using the symbol names
        given by keyword arguments.

        Args:
            t: time parameter
            gpu: whether to evaluate the array expressions on the GPU
            kwargs: a dictionary mapping parameter names to numeric arrays or
                    scalars

        Returns: a tuple of arrays corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        list_of_state_vectors = \
            [kwargs[symbol.name]
             for symbol
             in self.argument_symbols]

        return self.evaluate_with_list_of_state_vectors(
                        list_of_state_vectors=list_of_state_vectors,
                        t=t,
                        gpu=gpu)

    def evaluate(self,
                 state_matrices: tp.Tuple[NumericArrayOrScalar, ...],
                 t: tp.Optional[float] = None,
                 gpu: tp.Optional[bool] = None) \
            -> tp.Tuple[NumericArray, ...]:
        """
        This function evaluates the array expressions at arguments that are
        represented either as vectors or as horizontally concatenated vectors
        (i.e. arrays with 2 axes). It decomposes these matrices into column
        vectors and evaluates the array functions with this list of column
        vectors.

        The function returns a tuple of result arrays (one for each array
        function).

        Args:
            state_matrices: a tuple of matrices, vectors, or scalars
            t: time parameter
            gpu: whether to evaluate the array expressions on the GPU

        Returns: a tuple of arrays corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        if not isinstance(state_matrices, collections.abc.Sequence):
            if isinstance(state_matrices, (np.ndarray, cn.ndarray)):
                state_matrices = [state_matrices]
            else:
                raise TypeError("state_variables must be of type "
                                "Sequence[Union[numpy.ndarray, cocos.ndarray]]")

        list_of_state_vectors = []
        # If a state matrix has more than one axis, it is separated it into a
        # list of vectors and appended to the list of vector arguments.
        for state_matrix in state_matrices:
            if np.isscalar(state_matrix) or isinstance(state_matrix, numbers.Number):
                list_of_state_vectors.append(state_matrix)
            elif state_matrix.ndim > 1:
                for i in range(state_matrix.shape[1]):
                    list_of_state_vectors.append((state_matrix[:, i]))
            else:
                list_of_state_vectors.append(state_matrix)

        return self.evaluate_with_list_of_state_vectors(
                        list_of_state_vectors=list_of_state_vectors,
                        t=t,
                        gpu=gpu)


class LambdifiedMatrixExpressions(LambdifiedArrayExpressions):
    def __init__(
         self,
         symbolic_matrix_expressions: tp.Tuple[sym.MatrixBase, ...],
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         squeeze_column_vectors: tp.Optional[tp.Tuple[bool, ...]] = None,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: tp.Optional[tp.Tuple[bool, ...]] = None,
         dtype: np.generic = np.float32):

        if symbolic_time_function_name_to_numeric_time_function_map is None:
            symbolic_time_function_name_to_numeric_time_function_map = dict()

        super().__init__(
                    argument_symbols=argument_symbols,
                    time_symbol=time_symbol,
                    symbolic_array_expressions=symbolic_matrix_expressions,
                    symbolic_time_function_name_to_numeric_time_function_map
                    =symbolic_time_function_name_to_numeric_time_function_map,
                    squeeze_column_vectors=squeeze_column_vectors,
                    perform_cse=perform_cse,
                    lazy_initialization=lazy_initialization,
                    pre_attach=pre_attach,
                    dtype=dtype)

        rows = []
        cols = []

        for symbolic_matrix_expression in symbolic_matrix_expressions:
            r, c = symbolic_matrix_expression.shape
            rows.append(r)
            cols.append(c)

        self._rows = tuple(rows)
        self._cols = tuple(cols)

    @property
    def rows(self) -> tp.Tuple[int, ...]:
        return self._rows

    @property
    def cols(self) -> tp.Tuple[int, ...]:
        return self._cols

    @property
    def symbolic_matrix_expressions(self) -> tp.Tuple[sym.ImmutableMatrix, ...]:
        return self.symbolic_array_expressions


class LambdifiedVectorExpressions(LambdifiedMatrixExpressions):
    def __init__(
         self,
         symbolic_vector_expressions: tp.Tuple[sym.MatrixBase, ...],
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         squeeze_column_vectors: tp.Optional[tp.Tuple[bool, ...]] = None,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: tp.Optional[tp.Tuple[bool, ...]] = None,
         dtype: np.generic = np.float32):

        super().__init__(
                    symbolic_matrix_expressions=symbolic_vector_expressions,
                    argument_symbols=argument_symbols,
                    time_symbol=time_symbol,
                    symbolic_time_function_name_to_numeric_time_function_map
                    =symbolic_time_function_name_to_numeric_time_function_map,
                    squeeze_column_vectors=squeeze_column_vectors,
                    perform_cse=perform_cse,
                    lazy_initialization=lazy_initialization,
                    pre_attach=pre_attach, dtype=dtype)

    @property
    def dimension(self) -> int:
        return len(self.symbolic_vector_expressions)

    @property
    def symbolic_vector_expressions(self) -> tp.Tuple[sym.ImmutableMatrix, ]:
        return self.symbolic_matrix_expressions


################################################################################
# Single Lambdified Array, Matrix, and Vector-Expressions
################################################################################
class LambdifiedArrayExpression:
    def __init__(
         self,
         symbolic_array_expression: tp.Union[sym.MatrixBase, sym.Array],
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         squeeze_column_vector: bool = False,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: bool = True,
         dtype: np.generic = np.float32):

        self._lambdified_array_expressions = \
                LambdifiedArrayExpressions(
                    argument_symbols=argument_symbols,
                    time_symbol=time_symbol,
                    symbolic_array_expressions=(symbolic_array_expression, ),
                    symbolic_time_function_name_to_numeric_time_function_map
                    =symbolic_time_function_name_to_numeric_time_function_map,
                    squeeze_column_vectors=(squeeze_column_vector, ),
                    perform_cse=perform_cse,
                    lazy_initialization=lazy_initialization,
                    pre_attach=(pre_attach, ),
                    dtype=dtype)

    @property
    def is_column_vector(self) -> bool:
        return self._lambdified_array_expressions.is_column_vector(0)

    @property
    def is_row_vector(self) -> bool:
        return self._lambdified_array_expressions.is_row_vector(0)

    @property
    def argument_symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._lambdified_array_expressions.argument_symbols

    @property
    def number_of_state_variables(self) -> int:
        return self._lambdified_array_expressions.number_of_state_variables

    @property
    def time_symbol(self) -> sym.Symbol:
        return self._lambdified_array_expressions.time_symbol

    @property
    def symbolic_array_expression(self) -> sym.ImmutableMatrix:
        return self._lambdified_array_expressions.symbolic_array_expressions[0]

    @property
    def symbolic_time_function_name_to_numeric_time_function_map(self) \
            -> tp.Dict[str, tp.Callable]:
        return (self
                ._lambdified_array_expressions
                .symbolic_time_function_name_to_numeric_time_function_map)

    @property
    def symbols(self) -> tp.Tuple[sym.Symbol, ...]:
        return self._lambdified_array_expressions.symbols

    @property
    def number_of_variables(self) -> int:
        return self._lambdified_array_expressions.number_of_variables

    def evaluate_with_list_of_state_vectors(
            self,
            list_of_state_vectors: tp.Tuple[NumericArrayOrScalar, ...],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the array expression at arguments that are
        vectors (arrays with a single axis) or scalars and returns the resulting
        array.

        Args:
            list_of_state_vectors:
                A list of one-dimensional numeric arrays. The length of this
                list must match the number of arguments to the array-valued
                functions.

            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """

        return (self
                ._lambdified_array_expressions
                .evaluate_with_list_of_state_vectors(list_of_state_vectors,
                                                     t,
                                                     gpu)[0])

    def evaluate_with_dictionary(
            self,
            symbolic_to_numeric_parameter_map: tp.Dict[sym.Symbol,
                                                       NumericArrayOrScalar],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the array expression at arguments that are
        dictionaries mapping symbolic parameters to one-dimensional numeric
        arrays or scalars and returns the resulting array.

        Args:
            symbolic_to_numeric_parameter_map:
                A dictionary mapping symbolic parameters to numeric arrays or
                scalars.

            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """
        return (self
                ._lambdified_array_expressions
                .evaluate_with_dictionary(
                    symbolic_to_numeric_parameter_map
                    =symbolic_to_numeric_parameter_map,
                    t=t,
                    gpu=gpu))[0]

    def evaluate_with_kwargs(self,
                             t: tp.Optional[float] = None,
                             gpu: tp.Optional[bool] = None,
                             **kwargs) \
            -> NumericArray:
        """
        This function evaluates the array expression using the symbol names
        given by keyword arguments.

        Args:
            t: time parameter
            gpu: whether to evaluate the array expression on the GPU
            kwargs: a dictionary mapping parameter names to numeric arrays or
                    scalars

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """
        return (self
                ._lambdified_array_expressions
                .evaluate_with_kwargs(t=t,
                                      gpu=gpu,
                                      **kwargs)[0])

    def evaluate(self,
                 state_matrices: tp.Tuple[NumericArrayOrScalar, ...],
                 t: tp.Optional[float] = None,
                 gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the array expression at arguments that are
        represented either as vectors or as horizontally concatenated vectors
        (i.e. arrays with 2 axes). It decomposes these matrices into column
        vectors and evaluates the array functions with this list of column vectors.

        Args:
            state_matrices: a tuple of matrices, vectors, or scalars
            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        return (self
                ._lambdified_array_expressions
                .evaluate(state_matrices=state_matrices,
                          t=t,
                          gpu=gpu)[0])


class LambdifiedMatrixExpression(LambdifiedArrayExpression):
    def __init__(
         self,
         symbolic_matrix_expression: sym.MatrixBase,
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         squeeze_column_vector: bool = False,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: bool = True,
         dtype: np.generic = np.float32):

        super().__init__(argument_symbols=argument_symbols,
                         time_symbol=time_symbol,
                         symbolic_array_expression=symbolic_matrix_expression,
                         symbolic_time_function_name_to_numeric_time_function_map
                         =symbolic_time_function_name_to_numeric_time_function_map,
                         squeeze_column_vector=squeeze_column_vector,
                         perform_cse=perform_cse,
                         lazy_initialization=lazy_initialization,
                         pre_attach=pre_attach,
                         dtype=dtype)

    @property
    def rows(self) -> int:
        return (self
                ._lambdified_array_expressions
                .symbolic_array_expressions[0]
                .shape[0])

    @property
    def cols(self) -> int:
        return (self
                ._lambdified_array_expressions
                .symbolic_array_expressions[0].shape[1])

    @property
    def symbolic_matrix_expression(self) -> sym.ImmutableMatrix:
        return self.symbolic_array_expression


class LambdifiedVectorExpression(LambdifiedMatrixExpression):
    def __init__(
         self,
         symbolic_vector_expression: sym.MatrixBase,
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: bool = True,
         dtype: np.generic = np.float32):

        super().__init__(argument_symbols=argument_symbols,
                         time_symbol=time_symbol,
                         symbolic_matrix_expression=symbolic_vector_expression,
                         symbolic_time_function_name_to_numeric_time_function_map
                         =symbolic_time_function_name_to_numeric_time_function_map,
                         squeeze_column_vector=True,
                         perform_cse=perform_cse,
                         lazy_initialization=lazy_initialization,
                         pre_attach=pre_attach,
                         dtype=dtype)

    @property
    def symbolic_vector_expression(self) -> sym.ImmutableMatrix:
        return self.symbolic_matrix_expression


class LambdifiedScalarExpression(LambdifiedVectorExpression):
    def __init__(
         self,
         symbolic_expression: sym.Expr,
         argument_symbols: tp.Tuple[sym.Symbol, ...],
         time_symbol: tp.Optional[sym.Symbol] = None,
         symbolic_time_function_name_to_numeric_time_function_map:
         tp.Optional[tp.Dict[str, tp.Callable]] = None,
         perform_cse: bool = True,
         lazy_initialization: bool = False,
         pre_attach: bool = True,
         dtype: np.generic = np.float32):

        super().__init__(argument_symbols=argument_symbols,
                         time_symbol=time_symbol,
                         symbolic_vector_expression
                         =sym.ImmutableMatrix([symbolic_expression]),
                         symbolic_time_function_name_to_numeric_time_function_map
                         =symbolic_time_function_name_to_numeric_time_function_map,
                         perform_cse=perform_cse,
                         lazy_initialization=lazy_initialization,
                         pre_attach=pre_attach,
                         dtype=dtype)

    @property
    def symbolic_expression(self) -> sym.Expr:
        return self.symbolic_matrix_expression[0, 0]

    def evaluate_with_list_of_state_vectors(
            self,
            list_of_state_vectors: tp.Tuple[NumericArrayOrScalar, ...],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the scalar expression at arguments that are
        vectors (arrays with a single axis) or scalars and returns the resulting
        array.

        Args:
            list_of_state_vectors:
                A list of one-dimensional numeric arrays. The length of this
                list must match the number of arguments to the array-valued
                functions.

            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """

        return (super()
                .evaluate_with_list_of_state_vectors(
                    list_of_state_vectors=list_of_state_vectors,
                    t=t,
                    gpu=gpu)
                .squeeze())

    def evaluate_with_dictionary(
            self,
            symbolic_to_numeric_parameter_map: tp.Dict[sym.Symbol,
                                                       NumericArrayOrScalar],
            t: tp.Optional[float] = None,
            gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the scalar expression at arguments that are
        dictionaries mapping symbolic parameters to one-dimensional numeric
        arrays or scalars and returns the resulting array.

        Args:
            symbolic_to_numeric_parameter_map:
                A dictionary mapping symbolic parameters to numeric arrays or
                scalars.

            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """
        return (super()
                .evaluate_with_dictionary(symbolic_to_numeric_parameter_map
                                          =symbolic_to_numeric_parameter_map,
                                          t=t,
                                          gpu=gpu)
                .squeeze())

    def evaluate_with_kwargs(self,
                             t: tp.Optional[float] = None,
                             gpu: tp.Optional[bool] = None,
                             **kwargs) \
            -> NumericArray:
        """
        This function evaluates the scalar expression using the symbol names
        given by keyword arguments.

        Args:
            t: time parameter
            gpu: whether to evaluate the array expression on the GPU
            kwargs: a dictionary mapping parameter names to numeric arrays or
                    scalars

        Returns: an array corresponding to the array expressions evaluated at
                 the parameter vectors

        """
        return super().evaluate_with_kwargs(t=t, gpu=gpu, **kwargs).squeeze()

    def evaluate(self,
                 state_matrices: tp.Tuple[NumericArrayOrScalar, ...],
                 t: tp.Optional[float] = None,
                 gpu: tp.Optional[bool] = None) -> NumericArray:
        """
        This function evaluates the scalar expression at arguments that are
        represented either as vectors or as horizontally concatenated vectors
        (i.e. arrays with 2 axes). It decomposes these matrices into column
        vectors and evaluates the array functions with this list of column
        vectors.

        Args:
            state_matrices: a tuple of matrices, vectors, or scalars
            t: time parameter
            gpu: whether to evaluate the array expression on the GPU

        Returns: an array corresponding to the array expressions
                 evaluated at the parameter vectors

        """
        return (super()
                .evaluate(state_matrices=state_matrices,
                          t=t,
                          gpu=gpu)
                .squeeze())


def lambdify(args,
             expr,
             modules: tp.Optional[tp.Tuple] = None,
             printer: tp.Optional[tp.Tuple] = None,
             use_imps: bool = True,
             dummify: bool = True):
    if modules is None:
        modules = []

    return sym.lambdify(args=args,
                        expr=expr,
                        modules=list(modules) + [COCOS_TRANSLATIONS],
                        printer=printer,
                        use_imps=use_imps,
                        dummify=dummify)
