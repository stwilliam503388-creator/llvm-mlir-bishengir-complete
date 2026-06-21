// ==- toymini.cpp - Toy 语言解析器（从头实现）-==//
// 
// 目标：从零写 Toy 语言的前端，包括：
//   1. 词法分析器（Lexer）
//   2. 语法分析器（Parser）
//   3. AST 打印（展示结构）
//   4. MLIR 风格 IR 生成（文本形式，不依赖 LLVM）
//
// 对照 Toy Tutorial Ch1 + Ch2 的解析器部分。
// 不依赖任何 MLIR 库，纯 C++17 可编译运行。
//===

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <sstream>
#include <cctype>

// ============================================================
// 1. Token 类型定义
// ============================================================
enum class TokenKind {
  // 单字符
  LParen, RParen, LBrace, RBrace, LBracket, RBracket,
  Comma, Semicolon, Plus, Star, Equal, Minus,

  // 多字符
  Number, Ident, String,

  // 关键字
  Def, Var, Return, Print, Transpose,

  // 特殊
  Eof, Invalid
};

struct Token {
  TokenKind kind;
  std::string text;
  int line, col;
};

// ============================================================
// 2. Lexer（词法分析器）
// ============================================================
class Lexer {
  std::string input;
  int pos = 0, line = 1, col = 1;

  char peek() { return pos < (int)input.size() ? input[pos] : '\0'; }
  char advance() {
    char c = input[pos++];
    if (c == '\n') { line++; col = 1; } else { col++; }
    return c;
  }

  Token makeToken(TokenKind kind, std::string text) {
    return {kind, text, line, col - (int)text.size()};
  }

public:
  Lexer(std::string s) : input(std::move(s)) {}

  Token next() {
    while (isspace(peek())) advance();

    if (pos >= (int)input.size()) return makeToken(TokenKind::Eof, "");

    char c = peek();
    switch (c) {
      case '(': advance(); return makeToken(TokenKind::LParen, "(");
      case ')': advance(); return makeToken(TokenKind::RParen, ")");
      case '{': advance(); return makeToken(TokenKind::LBrace, "{");
      case '}': advance(); return makeToken(TokenKind::RBrace, "}");
      case '[': advance(); return makeToken(TokenKind::LBracket, "[");
      case ']': advance(); return makeToken(TokenKind::RBracket, "]");
      case ',': advance(); return makeToken(TokenKind::Comma, ",");
      case ';': advance(); return makeToken(TokenKind::Semicolon, ";");
      case '+': advance(); return makeToken(TokenKind::Plus, "+");
      case '*': advance(); return makeToken(TokenKind::Star, "*");
      case '=': advance(); return makeToken(TokenKind::Equal, "=");
      case '-': advance(); return makeToken(TokenKind::Minus, "-");
      default: break;
    }

    if (c == '#' || c == '/') {
      // 注释：跳过到行尾
      while (peek() != '\n' && peek() != '\0') advance();
      return next();
    }

    if (isalpha(c) || c == '_') {
      std::string text;
      while (isalnum(peek()) || peek() == '_') text += advance();
      if (text == "def")   return makeToken(TokenKind::Def, text);
      if (text == "var")   return makeToken(TokenKind::Var, text);
      if (text == "return") return makeToken(TokenKind::Return, text);
      if (text == "print") return makeToken(TokenKind::Print, text);
      if (text == "transpose") return makeToken(TokenKind::Transpose, text);
      return makeToken(TokenKind::Ident, text);
    }

    if (isdigit(c) || c == '.') {
      std::string text;
      while (isdigit(peek()) || peek() == '.') text += advance();
      return makeToken(TokenKind::Number, text);
    }

    return makeToken(TokenKind::Invalid, std::string(1, advance()));
  }
};

// ============================================================
// 3. AST 节点
// ============================================================
struct ASTNode {
  virtual ~ASTNode() = default;
  virtual std::string toString(int indent = 0) const = 0;
};

struct NumberExpr : ASTNode {
  double value;
  NumberExpr(double v) : value(v) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "Number(" + std::to_string(value) + ")";
  }
};

struct VariableExpr : ASTNode {
  std::string name;
  VariableExpr(std::string n) : name(std::move(n)) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "Var(" + name + ")";
  }
};

struct BinaryExpr : ASTNode {
  std::string op;
  std::unique_ptr<ASTNode> lhs, rhs;
  BinaryExpr(std::string o, std::unique_ptr<ASTNode> l, std::unique_ptr<ASTNode> r)
    : op(std::move(o)), lhs(std::move(l)), rhs(std::move(r)) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "Binary(" + op + ")\n" +
           lhs->toString(indent + 2) + "\n" + rhs->toString(indent + 2);
  }
};

struct CallExpr : ASTNode {
  std::string callee;
  std::vector<std::unique_ptr<ASTNode>> args;
  CallExpr(std::string c, auto a) : callee(std::move(c)), args(std::move(a)) {}
  std::string toString(int indent = 0) const override {
    std::string s = std::string(indent, ' ') + "Call(" + callee + ")";
    for (auto &a : args) s += "\n" + a->toString(indent + 2);
    return s;
  }
};

struct LiteralExpr : ASTNode {
  std::vector<std::vector<double>> values;
  LiteralExpr(std::vector<std::vector<double>> v) : values(std::move(v)) {}
  std::string toString(int indent = 0) const override {
    std::string s(indent, ' ');
    s += "Literal[" + std::to_string(values.size()) + "x";
    if (!values.empty()) s += std::to_string(values[0].size());
    else s += "0";
    s += "]";
    return s;
  }
};

struct VarDeclExpr : ASTNode {
  std::string name;
  std::unique_ptr<ASTNode> init;
  VarDeclExpr(std::string n, auto i) : name(std::move(n)), init(std::move(i)) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "VarDecl(" + name + ")\n" + init->toString(indent + 2);
  }
};

struct PrintExpr : ASTNode {
  std::unique_ptr<ASTNode> expr;
  PrintExpr(auto e) : expr(std::move(e)) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "Print\n" + expr->toString(indent + 2);
  }
};

struct ReturnExpr : ASTNode {
  std::unique_ptr<ASTNode> value;
  ReturnExpr(auto v) : value(std::move(v)) {}
  std::string toString(int indent = 0) const override {
    return std::string(indent, ' ') + "Return\n" + value->toString(indent + 2);
  }
};

struct Prototype {
  std::string name;
  std::vector<std::string> params;
  std::string toString(int indent = 0) const {
    std::string s = std::string(indent, ' ') + "Prototype(" + name + ", params=[";
    for (size_t i = 0; i < params.size(); i++) {
      if (i > 0) s += ", ";
      s += params[i];
    }
    return s + "])";
  }
};

struct Function {
  Prototype proto;
  std::vector<std::unique_ptr<ASTNode>> body;
  std::string toString(int indent = 0) const {
    std::string s = std::string(indent, ' ') + "Function(" + proto.name + ")\n";
    for (auto &b : body) s += b->toString(indent + 2) + "\n";
    return s;
  }
};

struct Module {
  std::vector<Function> functions;
};

// ============================================================
// 4. Parser（语法分析器）
// ============================================================
class Parser {
  Lexer &lexer;
  Token current;

  void advance() { current = lexer.next(); }
  bool check(TokenKind k) { return current.kind == k; }
  bool match(TokenKind k) { if (check(k)) { advance(); return true; } return false; }
  Token expect(TokenKind k, const std::string &msg) {
    if (check(k)) { auto t = current; advance(); return t; }
    std::cerr << "Error at " << current.line << ":" << current.col
              << ": expected " << msg << ", got '" << current.text << "'\n";
    exit(1);
  }

public:
  Parser(Lexer &l) : lexer(l) { advance(); }

  // 解析入口
  Module parseModule() {
    Module mod;
    while (!check(TokenKind::Eof))
      mod.functions.push_back(parseFunction());
    return mod;
  }

  // 函数定义: def name(params) { body }
  Function parseFunction() {
    expect(TokenKind::Def, "'def'");
    auto name = expect(TokenKind::Ident, "function name");
    expect(TokenKind::LParen, "'('");
    std::vector<std::string> params;
    while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
      params.push_back(expect(TokenKind::Ident, "parameter name").text);
      match(TokenKind::Comma);
    }
    expect(TokenKind::RParen, "')'");
    expect(TokenKind::LBrace, "'{'");
    std::vector<std::unique_ptr<ASTNode>> body;
    while (!check(TokenKind::RBrace) && !check(TokenKind::Eof))
      body.push_back(parseStatement());
    expect(TokenKind::RBrace, "'}'");
    return {Prototype{name.text, params}, std::move(body)};
  }

  // 语句
  std::unique_ptr<ASTNode> parseStatement() {
    if (check(TokenKind::Print)) {
      advance();
      expect(TokenKind::LParen, "'('");
      auto expr = parseExpr();
      expect(TokenKind::RParen, "')'");
      expect(TokenKind::Semicolon, "';'");
      return std::make_unique<PrintExpr>(std::move(expr));
    }
    if (check(TokenKind::Return)) {
      advance();
      auto expr = parseExpr();
      expect(TokenKind::Semicolon, "';'");
      return std::make_unique<ReturnExpr>(std::move(expr));
    }
    if (check(TokenKind::Var)) {
      advance();
      auto name = expect(TokenKind::Ident, "variable name").text;
      expect(TokenKind::Equal, "'='");
      auto init = parseExpr();
      expect(TokenKind::Semicolon, "';'");
      return std::make_unique<VarDeclExpr>(name, std::move(init));
    }
    auto expr = parseExpr();
    expect(TokenKind::Semicolon, "';'");
    return expr;
  }

  // 表达式（递归下降）
  std::unique_ptr<ASTNode> parseExpr() {
    return parseBinary(0);
  }

  // 优先级表
  int getPrecedence(const std::string &op) {
    if (op == "+" || op == "-") return 10;
    if (op == "*") return 20;
    return -1;
  }

  std::unique_ptr<ASTNode> parseBinary(int minPrec) {
    auto lhs = parsePrimary();
    while (true) {
      std::string op;
      if (check(TokenKind::Plus)) op = "+";
      else if (check(TokenKind::Star)) op = "*";
      else if (check(TokenKind::Minus)) op = "-";
      else break;

      int prec = getPrecedence(op);
      if (prec < minPrec) break;
      advance();
      auto rhs = parseBinary(prec + 1);
      lhs = std::make_unique<BinaryExpr>(op, std::move(lhs), std::move(rhs));
    }
    return lhs;
  }

  std::unique_ptr<ASTNode> parsePrimary() {
    if (check(TokenKind::Number)) {
      auto t = current;
      advance();
      return std::make_unique<NumberExpr>(std::stod(t.text));
    }
    if (check(TokenKind::Ident) || check(TokenKind::Transpose)) {
      auto t = current;
      advance();
      if (check(TokenKind::LParen)) {
        // 函数调用: transpose(...) or name(...)
        advance();  // eat (
        std::vector<std::unique_ptr<ASTNode>> args;
        while (!check(TokenKind::RParen) && !check(TokenKind::Eof)) {
          args.push_back(parseExpr());
          match(TokenKind::Comma);
        }
        expect(TokenKind::RParen, "')'");
        return std::make_unique<CallExpr>(t.text, std::move(args));
      }
      return std::make_unique<VariableExpr>(t.text);
    }
    if (check(TokenKind::LBracket)) {
      // 数组字面量: [[1,2],[3,4]]
      return parseLiteral();
    }
    if (check(TokenKind::LParen)) {
      advance();
      auto expr = parseExpr();
      expect(TokenKind::RParen, "')'");
      return expr;
    }
    std::cerr << "Error at " << current.line << ":" << current.col
              << ": unexpected token '" << current.text << "'\n";
    exit(1);
  }

  std::unique_ptr<ASTNode> parseLiteral() {
    expect(TokenKind::LBracket, "'['");
    std::vector<std::vector<double>> rows;
    while (!check(TokenKind::RBracket)) {
      expect(TokenKind::LBracket, "'[' for row");
      std::vector<double> row;
      while (!check(TokenKind::RBracket)) {
        auto t = expect(TokenKind::Number, "number");
        row.push_back(std::stod(t.text));
        match(TokenKind::Comma);
      }
      expect(TokenKind::RBracket, "']' for row");
      rows.push_back(std::move(row));
      match(TokenKind::Comma);
    }
    expect(TokenKind::RBracket, "']'");
    return std::make_unique<LiteralExpr>(std::move(rows));
  }
};

// ============================================================
// 5. MLIR 风格 IR 生成（文本输出，对照 Toy Tutorial Ch2）
// ============================================================
void genMLIR(const Module &mod) {
  std::cout << "module {\n";
  for (auto &fn : mod.functions) {
    std::cout << "  func.func @" << fn.proto.name << "(";
    for (size_t i = 0; i < fn.proto.params.size(); i++)
      std::cout << (i ? ", " : "") << "%" << fn.proto.params[i];
    std::cout << ") {\n";

    // 简单的变量表：模拟 MLIR 的 SSA 值
    std::map<std::string, int> varMap;
    int ssaCounter = 0;
    auto nextSSA = [&]() { return "%" + std::to_string(ssaCounter++); };

    for (auto &stmt : fn.body) {
      // 简单模式匹配生成 MLIR 文本
      if (auto varDecl = dynamic_cast<VarDeclExpr*>(stmt.get())) {
        auto result = nextSSA();
        varMap[varDecl->name] = ssaCounter - 1;
        std::cout << "    " << result << " = toy.constant { ... } : tensor<*xf64>\n";
      }
      // 其他语句简化处理
      std::cout << "    // " << stmt->toString() << "\n";
    }
    std::cout << "    toy.return\n";
    std::cout << "  }\n";
  }
  std::cout << "}\n";
}

// ============================================================
// main
// ============================================================
int main(int argc, char **argv) {
  std::string src = R"(
def main() {
  var A = [[1, 2, 3, 4], [5, 6, 7, 8]];
  var B = transpose(A);
  var C = B + A;
  print(C);
  return;
}
)";

  if (argc > 1) {
    // 从文件读取（简化，可以做）
    std::cout << "Usage: " << argv[0] << " [no args - uses embedded test]\n";
    return 0;
  }

  std::cout << "=== 源码 ===\n" << src << "\n";

  Lexer lexer(src);
  Parser parser(lexer);
  auto module = parser.parseModule();

  std::cout << "\n=== AST ===\n";
  for (auto &fn : module.functions)
    std::cout << fn.toString() << "\n";

  std::cout << "\n=== MLIR 风格 IR ===\n";
  genMLIR(module);

  return 0;
}
