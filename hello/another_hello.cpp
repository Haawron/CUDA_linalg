#include <iostream>
using namespace std;

int main() {
    int a, b;
    char op;
    cin >> a >> op >> b;
    cout << a << " " << op << " " << b << endl;
    cout << (int)(op == '/') << endl;
}