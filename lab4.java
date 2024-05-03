import java.util.*;
public class TicTacToe {
private char[] board;
private char currentPlayer;
public TicTacToe() {
board = new char[9];
Arrays.fill(board, ' ');
currentPlayer = 'X';
}
public void printBoard() {
for (int i = 0; i < 9; i += 3) {
System.out.println(board[i] + " | " + board[i + 1] + " | " +
board[i + 2]);
if (i < 6) System.out.println("---------");
}
System.out.println();
}
public List<Integer> availableMoves() {
List<Integer> moves = new ArrayList<>();
for (int i = 0; i < 9; i++) {
if (board[i] == ' ') {
moves.add(i);
}
}
return moves;
}
public boolean makeMove(int position) {
if (position < 0 || position >= 9 || board[position] != ' ') {
return false;
}
board[position] = currentPlayer;
currentPlayer = (currentPlayer == 'X') ? 'O' : 'X';
return true;
}
public char checkWinner() {
int[][] lines = {
{0, 1, 2}, {3, 4, 5}, {6, 7, 8},
{0, 3, 6}, {1, 4, 7}, {2, 5, 8},
{0, 4, 8}, {2, 4, 6}
};
for (int[] line : lines) {
if (board[line[0]] != ' ' && board[line[0]] == board[line[1]]
&& board[line[1]] == board[line[2]]) {
return board[line[0]];
}
}
if (availableMoves().isEmpty()) {
return 'T';
}
return ' ';
}
// MINIMAX algorithm
public char minimax(TicTacToe board, int depth, boolean
isMaximizingPlayer, int[] evaluatedNodes) {
char winner = board.checkWinner();
if (winner != ' ') {
return winner;
}
if (isMaximizingPlayer) {
char bestMove = 'O';
for (int move : board.availableMoves()) {
TicTacToe newBoard = new TicTacToe();
newBoard.board = Arrays.copyOf(board.board, 9);
newBoard.currentPlayer = board.currentPlayer;
newBoard.makeMove(move);
char score = minimax(newBoard, depth + 1, false,
evaluatedNodes);
evaluatedNodes[0]++;
if (score > bestMove) {
bestMove = score;
}
}
return bestMove;
} else {
char bestMove = 'X';
for (int move : board.availableMoves()) {
TicTacToe newBoard = new TicTacToe();
newBoard.board = Arrays.copyOf(board.board, 9);
newBoard.currentPlayer = board.currentPlayer;
newBoard.makeMove(move);
char score = minimax(newBoard, depth + 1, true,
evaluatedNodes);
evaluatedNodes[0]++;
if (score < bestMove) {
bestMove = score;
}
}
return bestMove;
}
}
// MINIMAX with alpha-beta pruning
public char alphabeta(TicTacToe board, int depth, boolean
isMaximizingPlayer, int alpha, int beta, int[] evaluatedNodes) {
char winner = board.checkWinner();
if (winner != ' ') {
return winner;
}
if (isMaximizingPlayer) {
char bestMove = 'O';
for (int move : board.availableMoves()) {
TicTacToe newBoard = new TicTacToe();
newBoard.board = Arrays.copyOf(board.board, 9);
newBoard.currentPlayer = board.currentPlayer;
newBoard.makeMove(move);
char score = alphabeta(newBoard, depth + 1, false, alpha,
beta, evaluatedNodes);
evaluatedNodes[0]++;
if (score > bestMove) {
bestMove = score;
}
alpha = Math.max(alpha, bestMove);
if (beta <= alpha) {
break;
}
}
return bestMove;
} else {
char bestMove = 'X';
for (int move : board.availableMoves()) {
TicTacToe newBoard = new TicTacToe();
newBoard.board = Arrays.copyOf(board.board, 9);
newBoard.currentPlayer = board.currentPlayer;
newBoard.makeMove(move);
char score = alphabeta(newBoard, depth + 1, true, alpha,
beta, evaluatedNodes);
evaluatedNodes[0]++;
if (score < bestMove) {
bestMove = score;
}
beta = Math.min(beta, bestMove);
if (beta <= alpha) {
break;
}
}
return bestMove;
}
}
public static void main(String[] args) {
TicTacToe game = new TicTacToe();
game.printBoard();
int[] minimaxNodes = new int[]{0};
game.minimax(game, 0, true, minimaxNodes);
System.out.println("Number of nodes evaluated for MINIMAX: " +
minimaxNodes[0]);
int[] alphabetaNodes = new int[]{0};
game.alphabeta(game, 0, true, Integer.MIN_VALUE,
Integer.MAX_VALUE, alphabetaNodes);
System.out.println("Number of nodes evaluated for ALPHA-BETA
PRUNING: " + alphabetaNodes[0]);
}
