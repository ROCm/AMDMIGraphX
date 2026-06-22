#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import cppcheck, itertools
from cppcheckdata import simpleMatch, match, getArguments


def skipTokenMatches(tokens, skip=None):
    for tok in tokens:
        if match(tok, skip):
            continue
        yield tok


def isTokensEqual(xtokens, ytokens, skip=None):
    for x, y in itertools.zip_longest(skipTokenMatches(xtokens, skip),
                                      skipTokenMatches(ytokens, skip)):
        if not x:
            return False
        if not y:
            return False
        if x.str != y.str:
            return False
    return True


def getInnerLink(token):
    if not token:
        return []
    if not token.link:
        return []
    return token.next.forward(token.link)


def getVariableDecl(var):
    if not var:
        return []
    end = var.typeEndToken
    if end:
        end = end.next
    return var.typeStartToken.forward(end)


def isFunctionCall(token):
    if not token:
        return False
    if not token.isName:
        return False
    if not token.next:
        return False
    if not token.next.str == '(':
        return False
    return True


# Conversion rank of the standard integral types, ordered from narrowest to
# widest, used to reason about the usual arithmetic conversions.
integralRank = {
    'bool': 0,
    'char': 1,
    'short': 2,
    'int': 3,
    'long': 4,
    'long long': 5
}

# Binary operators that apply the usual arithmetic conversions to both operands.
# This excludes the shift operators ('<<', '>>'), which promote each operand
# independently and take their result type from the left operand, so a cast of
# the right operand is not redundant there.
usualArithmeticConversionOps = {
    '+', '-', '*', '/', '%', '&', '|', '^', '<', '<=', '>', '>=', '==', '!='
}


def integralValueType(token):
    if not token:
        return None
    vt = token.valueType
    if not vt:
        return None
    if vt.pointer:
        return None
    if not vt.isIntegral():
        return None
    return vt


def integralTypeKey(token):
    # A (type, sign) key identifying a token's plain integral value type, or None
    # when it is not a non-pointer, non-reference integral.
    vt = integralValueType(token)
    if not vt or vt.reference in ('LValue', 'RValue'):
        return None
    return (vt.type, vt.sign)


def getStaticCast(token):
    # In the AST a cast is represented by the '(' token (isCast); for a
    # static_cast its first operand is the 'static_cast' keyword.
    if not token:
        return None
    if not token.isCast:
        return None
    if not simpleMatch(token.astOperand1, "static_cast"):
        return None
    return token


def staticCastOf(use):
    # Return the static_cast that has 'use' as its direct operand, or None.
    cast = getStaticCast(use.astParent)
    if cast and cast.astOperand2 is use:
        return cast
    return None


class TemplateBracketType:
    # Reads the type written inside a static_cast's template brackets as it
    # appears in the source. cppcheck merges fundamental types such as
    # 'unsigned int' into a single token in the simplified token list, so the
    # spelling is read from the raw tokens, which are indexed by position. The
    # index is built lazily on first use, since most files have no std::min or
    # std::max call with a static_cast argument at all.
    def __init__(self, data):
        self.data = data
        self.raw_by_pos = None

    def index(self):
        if self.raw_by_pos is None:
            self.raw_by_pos = {(tok.file, tok.linenr, tok.column): tok
                               for tok in self.data.rawTokens
                               if tok.str in ('<', '>')}
        return self.raw_by_pos

    def getRawTypeName(self, cast):
        # Use the simplified '<' link to bound the type, then read its spelling
        # from the raw '<' and '>' tokens looked up by position.
        lt = cast.astOperand1.next
        if not simpleMatch(lt, "<") or not lt.link:
            return None
        raw_by_pos = self.index()
        raw_lt = raw_by_pos.get((lt.file, lt.linenr, lt.column))
        raw_gt = raw_by_pos.get((lt.link.file, lt.link.linenr, lt.link.column))
        if not raw_lt or not raw_gt:
            return None
        parts = []
        for tok in raw_lt.next.forward(raw_gt):
            parts.append(tok.str)
        return " ".join(parts).replace(" :: ", "::")


@cppcheck.checker
def AvoidBranchingStatementAsLastInLoop(cfg, data):
    for token in cfg.tokenlist:
        if not token.str in ['for', 'while']:
            continue
        end = match(token, "for|while (*) {*}").end
        if not end:
            continue
        stmt = end.tokAt(-2)
        if not match(stmt, "%any% ; }"):
            continue
        if not match(stmt, "break|continue"):
            stmt = stmt.astTop()
        if match(stmt, "break|continue|return"):
            cppcheck.reportError(
                stmt, "style",
                "Branching statement as the last statement inside a loop is very confusing."
            )


# @cppcheck.checker
# def CollapsibleIfStatements(cfg, data):
#     for token in cfg.tokenlist:
#         if not match(token, "if (*) { if (*) {*} }"):
#             continue
#         cppcheck.reportError(token, "style", "These two if statements can be collapsed into one.")


@cppcheck.checker
def ConditionalAssert(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'if':
            continue
        if not match(token, "if (*) { assert (*) ; }"):
            continue
        cppcheck.reportError(token, "style",
                             "The if condition should be included in assert.")


@cppcheck.checker
def EmptyCatchStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'catch':
            continue
        if not match(token, "catch (*) { }"):
            continue
        cppcheck.reportError(token, "style", "An empty catch statement.")


@cppcheck.checker
def EmptyDoWhileStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'do':
            continue
        if not simpleMatch(token, "do { } while ("):
            continue
        cppcheck.reportError(token, "style", "Empty do-while.")


@cppcheck.checker
def EmptyElseBlock(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'else':
            continue
        if not simpleMatch(token, "else { }"):
            continue
        cppcheck.reportError(token, "style",
                             "Empty else statement can be safely removed.")


@cppcheck.checker
def EmptyForStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'for':
            continue
        if not match(token, "for (*) { }"):
            continue
        cppcheck.reportError(token, "style", "Empty for statement.")


@cppcheck.checker
def EmptyIfStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'if':
            continue
        if not match(token, "if (*) { }"):
            continue
        cppcheck.reportError(token, "style", "Empty if statement.")


@cppcheck.checker
def EmptySwitchStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'switch':
            continue
        if not match(token, "switch (*) { }"):
            continue
        cppcheck.reportError(token, "style", "Empty switch statement.")


@cppcheck.checker
def EmptyWhileStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'while':
            continue
        if not match(token, "while (*) { }"):
            continue
        cppcheck.reportError(token, "style", "Empty while statement.")


@cppcheck.checker
def ForLoopShouldBeWhileLoop(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'for':
            continue
        if not match(token, "for ( ; !!;"):
            continue
        # Skip empty for loops
        if match(token, "for (*) { }"):
            continue
        end = token.next.link
        if not match(end.tokAt(-1), "; )"):
            continue
        cppcheck.reportError(token, "style",
                             "For loop should be written as a while loop.")


@cppcheck.checker
def GotoStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'goto':
            continue
        cppcheck.reportError(token, "style", "Goto considered harmful.")


# @cppcheck.checker
# def InvertedLogic(cfg, data):
#     for token in cfg.tokenlist:
#         cond = None
#         if match(token, "if (*) {*} else { !!if"):
#             cond = token.next.astOperand2
#         elif match(token, "?"):
#             cond = token.astOperand1
#         if not match(cond, "!|!="):
#             continue
#         cppcheck.reportError(cond, "style", "It is cleaner to invert the logic.")


@cppcheck.checker
def LambdaAttribute(cfg, data):
    for token in data.rawTokens:
        if token.str != ']':
            continue
        if not match(token, "] __device__|__host__ {|("):
            continue
        cppcheck.reportError(
            token, "style",
            "Attributes to lambdas must come after parameter list.")


@cppcheck.checker
def MultipleUnaryOperator(cfg, data):
    for token in cfg.tokenlist:
        if not token.isUnaryOp(token.str):
            continue
        if not token.str in ['+', '-', '~', '!']:
            continue
        if not match(token.astOperand1, "+|-|~|!"):
            continue
        if not token.astOperand1.isUnaryOp(token.astOperand1.str):
            continue
        cppcheck.reportError(token, "style",
                             "Muliple unary operators used together.")


@cppcheck.checker
def MutableVariable(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'mutable':
            continue
        if not match(token, "mutable %name%"):
            continue
        cppcheck.reportError(token, "style",
                             "Do not create mutable variables.")


@cppcheck.checker
def NestedBlocks(cfg, data):
    for token in cfg.tokenlist:
        if not token.str in ['if', 'while', 'for', 'switch']:
            continue
        block = match(token, "if|while|for|switch (*) { {*}@block }").block
        if not block:
            block = match(token, "; { {*}@block break ; }").block
        if not block:
            continue
        cppcheck.reportError(block, "style", "Block directly inside block.")


@cppcheck.checker
def RedundantCast(cfg, data):
    for token in cfg.tokenlist:
        if not token.variable:
            continue
        m = match(token,
                  "%var%@decl ; %var%@assign = static_cast <*>@cast (*) ;")
        if m:
            if m.decl.varId != m.assign.varId:
                continue
        else:
            m = match(token, "%var%@assign = static_cast <*>@cast (*) ;")
            if not m:
                continue
            if m.assign.variable.typeEndToken != m.assign.previous:
                continue

        if not simpleMatch(token.previous, "auto"):
            if not isTokensEqual(getVariableDecl(m.assign.variable),
                                 getInnerLink(m.cast),
                                 skip='static|constexpr|const|volatile|&|&&|*'):
                continue
        cppcheck.reportError(token, "style", "Static cast is redundant.")


@cppcheck.checker
def RedundantConditionalOperator(cfg, data):
    for token in cfg.tokenlist:
        if token.str != '?':
            continue
        if not match(token, "? true|false : true|false"):
            continue
        cppcheck.reportError(token, "style",
                             "Conditional operator is redundant.")


@cppcheck.checker
def RedundantIfStatement(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'if':
            continue
        if not match(
                token,
                "if (*) { return true|false ; } else { return true|false ; }"):
            continue
        cppcheck.reportError(token, "style", "The if statement is redundant.")


@cppcheck.checker
def RedundantLocalVariable(cfg, data):
    for token in cfg.tokenlist:
        if not token.variable:
            continue
        m = match(token,
                  "%var%@decl ; %var%@assign = **; return %var%@returned ;")
        if not m:
            continue
        if m.decl.varId != m.assign.varId:
            continue
        if m.decl.varId != m.returned.varId:
            continue
        cppcheck.reportError(
            m.returned, "style",
            "Variable is returned immediately after its declaration, can be simplified to just return expression."
        )


@cppcheck.checker
def RedundantStaticCastOp(cfg, data):
    # Look for 'other <op> static_cast<T>(x)' (in either operand order) where the
    # binary operator applies the usual arithmetic conversions and 'other'
    # already has the cast's exact integral type. Those conversions then convert
    # the cast's source operand to that same type, so the explicit cast does not
    # change the result.
    for token in cfg.tokenlist:
        if token.str not in usualArithmeticConversionOps:
            continue
        op1 = token.astOperand1
        op2 = token.astOperand2
        # Both operands are required, which also rules out the unary forms of
        # operators such as '-', '&' and '*'.
        if not op1 or not op2:
            continue
        for cast_op, other in ((op1, op2), (op2, op1)):
            cast = getStaticCast(cast_op)
            if not cast or other.isCast:
                continue
            cast_vt = integralValueType(cast)
            other_vt = integralValueType(other)
            src_vt = integralValueType(cast.astOperand2)
            if not cast_vt or not other_vt or not src_vt:
                continue
            # Only consider genuine signed/unsigned integers, which excludes bool.
            if cast_vt.sign not in ('signed', 'unsigned'):
                continue
            # The other operand must already have the cast's exact integral type,
            # i.e. the same width and signedness.
            if cast_vt.type != other_vt.type or cast_vt.sign != other_vt.sign:
                continue
            src_rank = integralRank.get(src_vt.type)
            dst_rank = integralRank.get(cast_vt.type)
            if src_rank is None or dst_rank is None:
                continue
            # A narrowing cast (wider source than target) changes the value.
            if src_rank > dst_rank:
                continue
            # For a signed target an unsigned source can make the common type
            # unsigned, and whether the signed type can represent all of the
            # unsigned values is platform dependent, so the cast is not reliably
            # redundant.
            if cast_vt.sign == 'signed' and src_vt.sign == 'unsigned':
                continue
            cppcheck.reportError(
                cast.astOperand1, "style",
                "Static cast is redundant in a binary operation with an operand of the same type."
            )
            break


@cppcheck.checker
def StaticCastInMinMax(cfg, data):
    template_type = TemplateBracketType(data)
    for token in cfg.tokenlist:
        if token.str not in ('min', 'max'):
            continue
        # Only the plain 'std::min(...)' call form. The 'std::min<T>(...)' form
        # has '<' as the next token and is already what we recommend.
        if not match(token, "min|max ("):
            continue
        # Require the call to be qualified with 'std::' to avoid matching member
        # functions or other unrelated min/max overloads.
        if not simpleMatch(token.tokAt(-2), "std ::"):
            continue
        for arg in getArguments(token) or []:
            cast = getStaticCast(arg)
            if not cast:
                continue
            type_name = template_type.getRawTypeName(cast)
            if not type_name:
                continue
            cppcheck.reportError(
                token, "style",
                "Use 'std::%s<%s>' with an explicit template argument instead of a static_cast on an argument."
                % (token.str, type_name))
            break


@cppcheck.checker
def RedundantStaticCastDecl(cfg, data):
    # A local variable whose every read is 'static_cast<U>(x)' to the same type U
    # could be declared as U instead, removing all of those casts.
    uses_by_id = {}
    for tok in cfg.tokenlist:
        if tok.varId:
            uses_by_id.setdefault(tok.varId, []).append(tok)
    template_type = TemplateBracketType(data)
    for var in cfg.variables:
        if (not var.isLocal or var.isReference or var.isPointer or var.isArray
                or var.isStatic):
            continue
        nt = var.nameToken
        if not nt or not nt.varId:
            continue
        var_key = integralTypeKey(nt)
        if var_key is None:
            continue
        # The declaration is not a use. Its initializer either stays on the
        # declaration ('T x = ...') or, after simplification, is split off into a
        # following assignment ('T x ; x = ...'); skip the name and that
        # assignment so only genuine reads remain.
        skip = {nt}
        has_init = simpleMatch(nt.next, "=")
        split = match(nt, "%var%@decl ; %var%@assign =")
        if split and split.decl.varId == split.assign.varId:
            skip.add(split.assign)
            has_init = True
        if not has_init:
            continue
        reads = (u for u in uses_by_id.get(nt.varId, []) if u not in skip)
        casts = [staticCastOf(u) for u in reads]
        if not casts or not all(casts):
            continue
        # Every read must cast to the same integral type, and that type must
        # differ from the declaration (a same-type cast is handled elsewhere).
        cast_keys = {integralTypeKey(c) for c in casts}
        if None in cast_keys or len(cast_keys) != 1 or var_key in cast_keys:
            continue
        type_name = template_type.getRawTypeName(casts[0])
        if not type_name:
            continue
        cppcheck.reportError(
            nt, "style",
            "Variable '%s' is only used as a static_cast to '%s'; declare it as '%s' instead."
            % (nt.str, type_name, type_name))


# @cppcheck.checker
# def UnnecessaryElseStatement(cfg, data):
#     for token in cfg.tokenlist:
#         m = match(token, "if (*) {*}@block else@else_statement {")
#         if not m:
#             continue
#         stmt = m.block.link.tokAt(-2)
#         if not match(stmt, "%any% ; }"):
#             continue
#         if not match(stmt, "break|continue"):
#             stmt = stmt.astTop()
#         if not match(stmt, "break|continue|return|throw"):
#             continue
#         cppcheck.reportError(m.else_statement, "style", "Else statement is not necessary.")


@cppcheck.checker
def UnnecessaryEmptyCondition(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'if':
            continue
        m = match(token, "if (*)@if_cond { for (*)@for_cond {*} }")
        if not m:
            continue
        cond = m.if_cond.astOperand2
        if match(cond, "!"):
            cond = cond.astOperand1
        if not match(cond.tokAt(-2), ". empty ("):
            continue
        container = cond.tokAt(-2).astOperand1
        if not container.varId:
            continue
        if not match(m.for_cond.astOperand2, ":"):
            continue
        container_iter = m.for_cond.astOperand2.astOperand2
        if container_iter.varId != container.varId:
            continue
        cppcheck.reportError(
            container, "style",
            "Unnecessary check for empty before for range loop.")


@cppcheck.checker
def UseDeviceLaunch(cfg, data):
    for token in cfg.tokenlist:
        if not simpleMatch(token, "hipLaunchKernelGGL ("):
            continue
        cppcheck.reportError(token, "style", "Use device::launch instead.")


@cppcheck.checker
def UseManagePointer(cfg, data):
    functions = {
        "fclose", "free", "hipFree", "hipHostFree", "hipFreeArray",
        "hipMemFree", "hipStreamDestroy", "hipEventDestroy", "hipArrayDestroy",
        "hipCtxDestroy", "hipDestroyTextureObject", "hipDestroySurfaceObject",
        "miirDestroyHandle"
    }
    for token in cfg.tokenlist:
        if not isFunctionCall(token):
            continue
        if not token.str in functions:
            continue
        cppcheck.reportError(token, "style",
                             "Use manage pointer for resource management.")

@cppcheck.checker
def UseNamedLogicOperator(cfg, data):
    names = {'&&': 'and', '||': 'or', '!': 'not'}
    # cppcheck rewrites the alternative tokens 'and', 'or' and 'not' to their
    # symbolic spelling in the token list and does not always populate
    # originalName, so consult the raw token stream for operators that were
    # written in symbolic form -- anything else was already named.
    symbolic = {(tok.file, tok.linenr, tok.column)
                for tok in data.rawTokens if tok.str in names}
    if not symbolic:
        return
    for token in cfg.tokenlist:
        if token.str not in names:
            continue
        if (token.file, token.linenr, token.column) not in symbolic:
            continue
        if token.str == '&&':
            # A '&&' token can also be an rvalue reference ('T&&') rather than
            # a logical operator; require it to be a real expression.
            if not token.astParent:
                continue
            if not token.astOperand1:
                continue
            if not token.astOperand2:
                continue
            op1 = token.astOperand1
            op2 = token.astOperand2
            # An rvalue reference declared with a concrete type ('int&& x',
            # 'std::string&& x') has the declared variable on the right with
            # its valueType tagged as an RValue reference.
            if op2.valueType and op2.valueType.reference == 'RValue':
                continue
            # An rvalue reference whose type is a template parameter ('T&& x',
            # 'Ts&&... xs') has a bare type name with no varId as its left
            # operand.
            if op1.isName and not op1.varId and not op1.function:
                continue
        cppcheck.reportError(
            token, "style",
            "Use named logic operator '%s' for '%s'" % (names[token.str],
                                                       token.str))

@cppcheck.checker
def UseSmartPointer(cfg, data):
    for token in cfg.tokenlist:
        if token.str != 'new':
            continue
        if not match(token, "new %name%"):
            continue
        cppcheck.reportError(token, "style",
                             "Use manage pointer for resource management.")


@cppcheck.checker
def useStlAlgorithms(cfg, data):
    copy_functions = {"memcpy", "strcpy", "strncpy", "strcat", "strncat"}
    for token in cfg.tokenlist:
        if not isFunctionCall(token):
            continue
        if token.str in copy_functions:
            cppcheck.reportError(token, "style", "Use std::copy instead.")
        elif token.str == 'memset':
            cppcheck.reportError(token, "style", "Use std::fill instead.")
        elif token.str == 'memcmp':
            cppcheck.reportError(token, "style",
                                 "Use std::equal_range instead.")
        elif token.str == 'memchr':
            cppcheck.reportError(token, "style", "Use std::find instead.")


@cppcheck.checker
def MatcherNestedParentheses(cfg, data):
    for token in cfg.tokenlist:
        if not simpleMatch(token, "matcher ( ) const {"):
            continue
        for tok2 in token.tokAt(4).forward(token.linkAt(4)):
            if not simpleMatch(tok2, ") ) ) )"):
                continue
            if simpleMatch(tok2.link.previous, "bind"):
                continue
            cppcheck.reportError(
                tok2, "style",
                "Too many nested parentheses can affect readability; consider using variables instead."
            )
            break

@cppcheck.checker
def AvoidNestedValue(cfg, data):
    for token in cfg.tokenlist:
        if not simpleMatch(token, ":: value"):
            continue
        parent = token.previous
        if not match(parent, ">|)"):
            continue
        valueTok = token.next
        if valueTok.typeScope != None:
            continue
        if match(valueTok.next, "::|("):
            continue
        cppcheck.reportError(
            token, "style",
            "Construct trait instead of using ::value"
        )

@cppcheck.checker
def UseCachedEnvVar(cfg, data):
    env_functions = {"enabled", "disabled", "env", "value_of", "string_value_of"}
    for token in cfg.tokenlist:
        if not isFunctionCall(token):
            continue
        if not token.str in env_functions:
            continue
        m = match(token, "%name% ( %name%@env_var :: value ( ) )|,")
        if not m:
            continue
        if(m.env_var.str == "T"):
            continue
        cppcheck.reportError(
            token, "style",
            "Use cached env variable"
        )
