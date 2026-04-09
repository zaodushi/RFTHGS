import subprocess
import tempfile
import json
import os
import re
import joblib

def remove_headers(code):
    """移除所有头文件包含、宏定义和pragma指令"""
    lines = code.split('\n')
    clean_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 移除 #include 语句（无论是尖括号还是双引号）
        if stripped.startswith('#include'):
            continue
        # 移除 #define 宏定义
        if stripped.startswith('#define'):
            continue
        # 移除 #pragma 指令
        if stripped.startswith('#pragma'):
            continue
        # 移除单行注释
        if stripped.startswith('//'):
            continue
        clean_lines.append(line)
    
    return '\n'.join(clean_lines)

def normalize_type_string(type_str):
    """标准化类型字符串，移除临时文件路径等不相关信息"""
    if not type_str:
        return type_str
    
    # 使用正则表达式移除临时文件路径信息
    # 匹配模式：(lambda at /path/to/temp_file.cpp:line:column)
    normalized = re.sub(r'\(lambda at [^)]+\)', '(lambda)', type_str)
    
    # 移除具体的临时文件路径，保留lambda的基本结构
    normalized = re.sub(r'/tmp/[^:]+\.cpp:\d+:\d+', 'TEMP_FILE', normalized)
    
    return normalized

def is_duplicate_cpp_code(code1: str, code2: str, include_dirs=None) -> bool:

    """
    判断两份C++代码的AST是否完全相同
    使用Clang构建AST，并进行标准化处理后比较
    
    Args:
        code1: 第一份C++代码
        code2: 第二份C++代码  
        include_dirs: 头文件搜索路径列表，默认为None
    """
    def extract_function_ast(ast_data):
        """提取函数定义的AST，忽略系统内置声明"""
        if not isinstance(ast_data, dict):
            return []
            
        functions = []
        
        def process_node(node):
            if not isinstance(node, dict):
                return
                
            kind = node.get('kind')
            if kind == 'FunctionDecl':
                # 跳过没有函数体的声明
                has_body = any(child.get('kind') == 'CompoundStmt' 
                             for child in node.get('inner', []))
                if not has_body:
                    return
                    
                func_info = {
                    'kind': 'FunctionDecl',
                    'name': node.get('name', 'unnamed_function'),
                    'type': normalize_type_string(node.get('type', {}).get('qualType', '')),
                    'params': [],
                    'body': None
                }
                
                # 提取参数
                for child in node.get('inner', []):
                    if isinstance(child, dict) and child.get('kind') == 'ParmVarDecl':
                        func_info['params'].append({
                            'name': child.get('name', 'unnamed_param'),
                            'type': normalize_type_string(child.get('type', {}).get('qualType', ''))
                        })
                
                # 提取函数体
                for child in node.get('inner', []):
                    if isinstance(child, dict) and child.get('kind') == 'CompoundStmt':
                        func_info['body'] = simplify_stmt(child)
                        break
                
                functions.append(func_info)
            
            # 递归处理子节点
            for child in node.get('inner', []):
                if isinstance(child, dict):
                    process_node(child)
        
        process_node(ast_data)
        return functions
    
    def simplify_stmt(stmt):
        """简化语句结构，只保留关键信息"""
        if not isinstance(stmt, dict):
            return stmt
        
        kind = stmt.get('kind')
        
        if kind == 'ReturnStmt':
            # 简化return语句
            inner = stmt.get('inner', [])
            if inner:
                return {
                    'kind': 'ReturnStmt',
                    'value': simplify_expr(inner[0]) if inner else None
                }
        
        elif kind == 'BinaryOperator':
            # 简化二元操作
            inner = stmt.get('inner', [])
            if len(inner) >= 2:
                return {
                    'kind': 'BinaryOperator',
                    'opcode': stmt.get('opcode', ''),
                    'lhs': simplify_expr(inner[0]),
                    'rhs': simplify_expr(inner[1]) if len(inner) > 1 else None
                }
        
        elif kind == 'CompoundStmt':
            # 简化复合语句
            statements = []
            for child in stmt.get('inner', []):
                simplified = simplify_stmt(child)
                if simplified:
                    statements.append(simplified)
            return {
                'kind': 'CompoundStmt',
                'statements': statements
            }
        
        elif kind == 'DeclStmt':
            # 变量声明语句
            return {
                'kind': 'DeclStmt',
                'declarations': [simplify_stmt(child) for child in stmt.get('inner', []) 
                               if isinstance(child, dict)]
            }
        
        elif kind == 'IfStmt':
            # if语句
            inner = stmt.get('inner', [])
            return {
                'kind': 'IfStmt',
                'condition': simplify_expr(inner[0]) if inner else None,
                'then': simplify_stmt(inner[1]) if len(inner) > 1 else None,
                'else': simplify_stmt(inner[2]) if len(inner) > 2 else None
            }
        
        return {'kind': kind} if kind else stmt
    
    def simplify_expr(expr):
        """简化表达式"""
        if not isinstance(expr, dict):
            return expr
        
        kind = expr.get('kind')
        
        if kind == 'DeclRefExpr':
            # 变量引用
            return {
                'kind': 'DeclRefExpr',
                'name': expr.get('name', ''),
                'type': normalize_type_string(expr.get('type', {}).get('qualType', ''))
            }
        
        elif kind == 'IntegerLiteral':
            # 整数字面量
            return {
                'kind': 'IntegerLiteral',
                'value': 'int_literal',  # 只保留类型，忽略具体值
                'type': normalize_type_string(expr.get('type', {}).get('qualType', 'int'))
            }
        
        elif kind == 'BinaryOperator':
            # 二元操作
            inner = expr.get('inner', [])
            if len(inner) >= 2:
                return {
                    'kind': 'BinaryOperator',
                    'opcode': expr.get('opcode', ''),
                    'lhs': simplify_expr(inner[0]),
                    'rhs': simplify_expr(inner[1]) if len(inner) > 1 else None
                }
        
        elif kind == 'ImplicitCastExpr':
            # 隐式转换
            inner = expr.get('inner', [])
            if inner:
                return simplify_expr(inner[0])
        
        elif kind == 'CallExpr':
            # 函数调用
            inner = expr.get('inner', [])
            return {
                'kind': 'CallExpr',
                'callee': simplify_expr(inner[0]) if inner else None,
                'args': [simplify_expr(arg) for arg in inner[1:]] if len(inner) > 1 else []
            }
        
        elif kind == 'MemberExpr':
            # 成员访问
            inner = expr.get('inner', [])
            return {
                'kind': 'MemberExpr',
                'base': simplify_expr(inner[0]) if inner else None,
                'member': expr.get('name', '')
            }
        
        elif kind == 'StringLiteral':
            # 字符串字面量
            return {
                'kind': 'StringLiteral',
                'type': normalize_type_string(expr.get('type', {}).get('qualType', ''))
            }
        
        elif kind == 'CXXStaticCastExpr':
            # C++静态转换表达式，主要关注转换的目标类型和被转换的表达式
            inner = expr.get('inner', [])
            return {
                'kind': 'CXXStaticCastExpr',
                'target_type': normalize_type_string(expr.get('type', {}).get('qualType', '')),
                'operand': simplify_expr(inner[0]) if inner else None
            }
        
        return {'kind': kind} if kind else expr
    
    def alpha_rename_functions(functions):
        """对函数和参数进行alpha-renaming"""
        name_map = {}
        counter = 0
        
        for func in functions:
            # 重命名函数
            old_name = func['name']
            new_name = f'func_{counter}'
            name_map[old_name] = new_name
            func['name'] = new_name
            counter += 1
            
            # 重命名参数
            param_counter = 0
            for param in func['params']:
                old_param = param['name']
                new_param = f'param_{param_counter}'
                name_map[old_param] = new_param
                param['name'] = new_param
                param_counter += 1
        
        # 在函数体中应用重命名
        def rename_in_expr(expr):
            if isinstance(expr, dict):
                if expr.get('kind') == 'DeclRefExpr' and expr.get('name') in name_map:
                    expr['name'] = name_map[expr['name']]
                
                for key, value in expr.items():
                    if isinstance(value, dict):
                        rename_in_expr(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                rename_in_expr(item)
        
        for func in functions:
            if func.get('body'):
                rename_in_expr(func['body'])
    
    def get_ast_json(code, include_dirs=None):
        """使用Clang获取代码的AST，支持C++17标准"""
        if code is None:
            print("Warning: Input code for AST generation is None.")
            return None
            
        include_dirs = include_dirs or []
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        try:
            # 构建Clang命令，使用C++17标准
            cmd = [
                'clang', 
                '-Xclang', 
                '-ast-dump=json',
                '-w', '-fno-color-diagnostics',
                '-fno-implicit-templates',     
                '-std=c++20',  # 添加C++17标准
                '-fsyntax-only',
                temp_file
            ]
            
            # 添加头文件搜索路径
            for include_dir in include_dirs:
                cmd.extend(['-I', include_dir])
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Clang error: {result.stderr}")
                return None
                
            ast_data = json.loads(result.stdout)
            functions = extract_function_ast(ast_data)
            alpha_rename_functions(functions)
            return functions
            
        except subprocess.TimeoutExpired:
            print("Clang processing timeout")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"Error processing with Clang: {e}")
            return None
        finally:
            try:
                os.unlink(temp_file)
            except (PermissionError, OSError):
                pass
    
    # 预处理代码，移除头文件等
    # clean_code1 = remove_headers(code1)
    # clean_code2 = remove_headers(code2)
    
    # 获取两份代码的AST
    #ast1 = get_ast_json(code1, include_dirs)
    #ast2 = get_ast_json(code2, include_dirs)
    tasks = [
        (code1, include_dirs),
        (code2, include_dirs)
    ]
    results = joblib.Parallel(n_jobs=2)(
        joblib.delayed(get_ast_json)(code, include_dirs)
        for code, include_dirs in tasks
    )
    ast1, ast2 = results
    
    if ast1 is None or ast2 is None:
        return False
    
    # 调试信息：打印提取的AST
    # if __name__ == "__main__":
    #     print("AST1:", json.dumps(ast1, indent=2))
    #     print("AST2:", json.dumps(ast2, indent=2))
    
    # 比较标准化后的AST
    result = json.dumps(ast1, sort_keys=True) == json.dumps(ast2, sort_keys=True)
    # 如果不一样，打印不一致的地方 
    return result

# 测试函数
def test_with_headers():
    """测试头文件搜索路径功能"""
    # 示例：假设头文件在./include目录下
    code1 = '''
    double a;
    int b;
    int add(int a, int b) { return a + b; }
    '''
    
    code2 = '''
    int a;
    double b;
    int add(int x, int y) { return y * x; }
    '''
    
    # 测试时指定头文件目录
    include_dirs = ["./include", "./headers"]
    result = is_duplicate_cpp_code(code1, code2, include_dirs)
    print("带头文件测试:", result)

if __name__ == "__main__":
    code = """

#include "selective_route_exchange.h"

#include "DynamicBitset.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

using Client  = size_t;
using Clients = std::vector<Client>;
using Route   = pyvrp::Route;
using Routes  = std::vector<Route>;

namespace pyvrp::crossover
{
/*
 * Minimal selective-route-exchange:
 * 1. Take numMovedRoutes consecutive routes from parent B (starting index given by startIndices.second).
 * 2. Remove the clients of these routes from parent A's routes, then append the entire routes to child 1.
 * 3. Do the symmetric operation to obtain child 2.
 * 4. Evaluate penalised costs and return the better one.
 */
pyvrp::Solution selectiveRouteExchange(
    std::pair<Solution const *, Solution const *> const &parents,
    ProblemData const &data,
    CostEvaluator const &costEvaluator,
    std::pair<size_t, size_t> const &startIndices,
    size_t const numMovedRoutes)
{
    auto const *parentA = parents.first;
    auto const *parentB = parents.second;

    size_t const nRoutesA = parentA->numRoutes();
    size_t const nRoutesB = parentB->numRoutes();

    if (numMovedRoutes == 0 || numMovedRoutes > std::min(nRoutesA, nRoutesB))
        throw std::invalid_argument("numMovedRoutes out of range.");

    size_t const startA = startIndices.first  % nRoutesA;
    size_t const startB = startIndices.second % nRoutesB;

    /* ------ helper lambda that builds a single child ------ */
    auto makeChild = [&](Routes const &donorRoutes,
                         Routes const &receiverRoutes,
                         size_t donorStartIdx) -> Solution
    {
        /* 1) Take numMovedRoutes consecutive routes from the donor */
        Routes moved;
        moved.reserve(numMovedRoutes);
        for (size_t r = 0; r < numMovedRoutes; ++r)
            moved.push_back(donorRoutes[(donorStartIdx + r) % donorRoutes.size()]);

        /* 2) Mark all clients of these routes in a DynamicBitset */
        DynamicBitset mark(data.numLocations());
        for (Route const &route : moved)
            for (Client c : route)
                mark[c] = true;             // set bit

        /* 3) Copy receiver's routes and delete marked clients */
        Routes childRoutes;
        childRoutes.reserve(receiverRoutes.size() + moved.size());

        for (Route const &route : receiverRoutes)
        {
            Clients kept;
            kept.reserve(route.size());

            for (Client c : route)
                if (!mark[c])
                    kept.push_back(c);

            if (!kept.empty())
                childRoutes.emplace_back(data, kept, route.vehicleType());
        }

        /* 4) Append the donor routes to the end */
        for (Route const &route : moved)
            childRoutes.emplace_back(data,
                                     Clients(route.begin(), route.end()),
                                     route.vehicleType());

        return Solution(data, childRoutes);
    };

    /* ------ build two children and return the best ------ */
    Solution child1 = makeChild(parentB->routes(), parentA->routes(), startB);
    Solution child2 = makeChild(parentA->routes(), parentB->routes(), startA);

    return costEvaluator.penalisedCost(child1) <
           costEvaluator.penalisedCost(child2)
           ? child1
           : child2;
}
}  // namespace pyvrp::crossover
            """
    import time
    start_time = time.time()
    r = is_duplicate_cpp_code(code,'\n' + code + '\n  ', include_dirs=[
                        '../pyvrp/pyvrp/cpp',
                        '../pyvrp/pyvrp/cpp/crossover'
                    ])
    print(r)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


    