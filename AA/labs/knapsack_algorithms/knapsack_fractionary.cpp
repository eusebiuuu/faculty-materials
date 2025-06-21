#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, max_w;
    cin >> n >> max_w;
    vector<int> vals(n + 1), weights(n + 1);

    for (int i = 1; i <= n; ++i) {
        cin >> vals[i];
    }
    for (int i = 1; i <= n; ++i) {
        cin >> weights[i];
    }

    vector<pair<double, int>> ratios(n + 1);
    for (int i = 1; i <= n; ++i) {
        ratios[i].first = (double) vals[i] / weights[i];
        ratios[i].second = i;
    }
    
    sort(ratios.begin() + 1, ratios.end(), greater<pair<double, int>>());
    double ans = 0;
    for (int i = 1; i <= n; ++i) {
        int curr_w = weights[ratios[i].second];
        if (curr_w <= max_w) {
            max_w -= curr_w;
            ans += vals[ratios[i].second];
        } else {
            ans += ratios[i].first * max_w;
            break;
        }
    }

    cout << ans << '\n';
    return 0;
}
