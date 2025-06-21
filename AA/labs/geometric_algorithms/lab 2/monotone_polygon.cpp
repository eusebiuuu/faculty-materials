#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;
int const INF = 1e9 + 9, N = 1e6;
int points[2][N];

bool is_monotonical(int n, int idx) {
    int start = 0, min_coord = INF;
    for (int i = 0; i < n; ++i) {
        if (points[idx][i] < min_coord) {
            min_coord = points[idx][i];
            start = i;
        }
    }

    if (points[idx][0] == points[idx][n - 1]) {
        return true;
    }

    int l = (start + 1) % n, r = (start - 1 + n) % n;
    while (points[idx][l] >= points[idx][(l + n - 1) % n]) {
        l++;
    }

    while (points[idx][r] >= points[idx][(r + 1) % n]) {
        r = (r - 1 + n) % n;
    }

    return points[idx][(l + n - 1) % n] == points[idx][(r + 1) % n] && l > r;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2; ++j) {
            cin >> points[j][i];
        }
    }

    cout << (is_monotonical(n, 0) ? "YES" : "NO") << '\n';
    cout << (is_monotonical(n, 1) ? "YES" : "NO") << '\n';

    return 0;
}
