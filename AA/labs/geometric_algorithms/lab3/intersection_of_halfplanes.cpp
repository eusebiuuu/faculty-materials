#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;
int const INF = 1e9;

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    double min_x = -2 * INF, max_x = 2 * INF;
    double min_y = -2 * INF, max_y = 2 * INF;
    for (int i = 1; i <= n; ++i) {
        int x, y, c;
        cin >> x >> y >> c;
        if (x == 0) {
            if (y > 0) {
                max_y = min(max_y, -1.0 * c / y);
            } else {
                min_y = max(min_y, -1.0 * c / y);
            }
        } else {
            if (x > 0) {
                max_x = min(max_x, -1.0 * c / x);
            } else {
                min_x = max(min_x, -1.0 * c / x);
            }
        }
    }

    // cout << min_x << ' ' << max_x << ' ' << min_y << ' ' << max_y << '\n';

    if (min_x > max_x || min_y > max_y) {
        return cout << "VOID\n", 0;
    }

    if (min_x <= max_x && min_y <= max_y && min_x > -INF && max_x < INF && min_y > -INF && max_y < INF) {
        return cout << "BOUNDED\n", 0;
    }

    return cout << "UNBOUNDED\n", 0;
}
