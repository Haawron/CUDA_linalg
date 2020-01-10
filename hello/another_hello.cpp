#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
using namespace std;

// void func(int (*p)[3]) {
//     for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
//         cout << p[i][j] << (j == 2 ? "\n" : " ");
// }

// void square(const int *x) {
//     for (int i = 0; i < 5; i++) x[i] *= x[i];
// }

int main() {

    const int N = 2300;
    int data[N], cnts[10] = {0};
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 10;
        cnts[data[i]]++;
    }

    for (int i = 0; i < N; i++) cout << data[i] << " ";
    cout << endl << endl;

    for (int j = 0; j < 10; j++) {
        cout << j << " ";
        printf("%6.3f %% ", cnts[j] / (double)N * 100);
        for (int k = 0; k < cnts[j]; k++)
            cout << "*";
        cout << endl;
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    // const int N = 100;
    // char ch[N];
    // for (int i = 0; i < N; i++) ch[i] = rand() % ('Z' - 'A' + 1) + 'A';
    // for (int i = 0; i < N; i++) cout << ch[i];
    // cout << endl;
    // bool swapped;
    // while (true) {
    //     swapped = false;
    //     for (int i = 0; i < N - 1; i++)
    //         if (ch[i] < ch[i + 1]) {
    //             swap(ch[i], ch[i + 1]);
    //             swapped = true;
    //         }
    //     if (!swapped) break;
    // }
    // for (int i = 0; i < N; i++) cout << ch[i];
    // cout << endl;

    ////////////////////////////////////////////////////////////////////////////////////////

    // const int N = 10000;
    // const int br = 30;
    // int x[N];
    // for (int i = 0; i < N; i++) x[i] = rand() % (30 * N);

    // for (int i = 0; i < N; i++)
    //     cout << setw(6) << x[i] << (i % br == (br - 1) ? "\n" : " ");
    // cout << endl << endl << endl;

    // bool swapped;
    // while (true) {
    //     swapped = false;
    //     for (int i = 0; i < N - 1; i++)
    //         if (x[i] < x[i + 1]) {
    //             swap(x[i], x[i + 1]);
    //             swapped = true;
    //         }
    //     if (!swapped) break;
    // }

    // for (int i = 0; i < N; i++)
    //     cout << setw(6) << x[i] << (i % br == (br - 1) ? "\n" : " ");
}
