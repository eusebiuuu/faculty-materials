#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, max_w;
    cin >> n >> max_w;
    vector<int> max_val(max_w + 1), vals(n + 1), weights(n + 1);

    for (int i = 1; i <= n; ++i) {
        cin >> vals[i];
    }
    for (int i = 1; i <= n; ++i) {
        cin >> weights[i];
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = max_w; j >= weights[i]; --j) {
            max_val[j] = max(max_val[j - weights[i]] + vals[i], max_val[j]);
        }
    }

    int ans = 0;
    for (int i = 1; i <= max_w; ++i) {
        ans = max(ans, max_val[i]);
    }
    cout << ans << '\n';
    return 0;
}
