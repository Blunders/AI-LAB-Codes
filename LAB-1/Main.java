
//Q.5 Solution for the puzzle problem

import java.util.*;
class PuzzleState {
private final int[][] state;
public PuzzleState(int[][] state) {
this.state = state;
}
public int[][] getState() {
return state;
}
@Override
public int hashCode() {
return Arrays.deepHashCode(state);
}
@Override
public boolean equals(Object obj) {
if (this == obj) {
return true;
}
if (obj == null || getClass() != obj.getClass()) {
return false;
}
PuzzleState otherState = (PuzzleState) obj;
return Arrays.deepEquals(this.state, otherState.state);
}
}
public class Main {
static HashSet<PuzzleState> set = new HashSet<>();
public static void main(String[] args) {
int depth = 2; // Set the desired depth
List<int[][]> instances = generateSolvableInstances(depth);
System.out.println("Instances at Depth " + depth + ":");
for (int i = 0; i < instances.size(); i++) {
System.out.println("Instance " + (i + 1) + ":");
printBoard(instances.get(i));
System.out.println();
}
}
public static List<int[][]> generateSolvableInstances(int depth) {
List<int[][]> instances = new ArrayList<>();
int[][] goalState = {
{1, 2, 3},
{4, 5, 6},
{7, 8, 0}
};
set.add(new PuzzleState(goalState));
generateInstancesRecursively(new PuzzleState(goalState), depth,
instances);
return instances;
}
public static void generateInstancesRecursively(PuzzleState
currentState, int depth, List<int[][]> instances) {
if (depth == 0) {
instances.add(deepCopy(currentState.getState()));
return;
}
List<int[]> legalMoves =
getLegalMoves(currentState.getState());
Collections.shuffle(legalMoves);
for (int[] move : legalMoves) {
int[][] nextState = deepCopy(currentState.getState());
swap(nextState, move[0], move[1], move[2], move[3]);
PuzzleState nextPuzzleState = new PuzzleState(nextState);
if (set.contains(nextPuzzleState)) {
continue;
}
set.add(nextPuzzleState);
generateInstancesRecursively(nextPuzzleState, depth-1,
instances);
}
}
public static List<int[]> getLegalMoves(int[][] board) {
List<int[]> legalMoves = new ArrayList<>();
int emptyRow =-1;
int emptyCol =-1;
// Find the position of the empty space (0)
outerLoop:
for (int i = 0; i < board.length; i++) {
for (int j = 0; j < board[0].length; j++) {
if (board[i][j] == 0) {
emptyRow = i;
emptyCol = j;
break outerLoop;
}
}
}
// Check legal moves (up, down, left, right)
int[][] moves = {
{-1, 0}, // up
{1, 0}, // down
{0,-1}, // left
{0, 1} // right
};
for (int[] move : moves) {
int newRow = emptyRow + move[0];
int newCol = emptyCol + move[1];
if (isValidMove(newRow, newCol)) {
legalMoves.add(new int[]{emptyRow, emptyCol, newRow,
newCol});
}
}
return legalMoves;
}
public static boolean isValidMove(int row, int col) {
return row >= 0 && row < 3 && col >= 0 && col < 3;
}
private static void swap(int[][] board, int row1, int col1, int
row2, int col2) {
int temp = board[row1][col1];
board[row1][col1] = board[row2][col2];
board[row2][col2] = temp;
}
private static int[][] deepCopy(int[][] original) {
int[][] copy = new int[original.length][];
for (int i = 0; i < original.length; i++) {
copy[i] = original[i].clone();
}
return copy;
}
private static void printBoard(int[][] board) {
for (int[] row : board) {
for (int cell : row) {
System.out.print(cell + " ");
}
System.out.println();
}
}
}