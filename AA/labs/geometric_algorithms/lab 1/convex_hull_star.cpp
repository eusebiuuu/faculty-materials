#include <bits/stdc++.h>
#define ll long long
#define ii pair<ll, ll>
using namespace std;

ll get_orientation_value(ii p1, ii p2, ii p3) {
    ll sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    ll sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n, min_x = 1e9, min_idx = 0;
    cin >> n;
    vector<ii> points(n), convex_hull;

    for (int i = 0; i < n; ++i) {
        cin >> points[i].first >> points[i].second;
        if (points[i].first < min_x) {
            min_idx = i;
            min_x = points[i].first;
        }
    }

    convex_hull.push_back(points[min_idx]);
    convex_hull.push_back(points[(min_idx + 1) % n]);

    for (int i = 2; i <= n; ++i) {
        int curr_idx = (min_idx + i) % n;

        while (convex_hull.size() > 1) {
            ll orientation = get_orientation_value(convex_hull[convex_hull.size() - 2], convex_hull.back(), points[curr_idx]);
            if (orientation > 0) {
                break;
            }
            convex_hull.pop_back();
        }

        convex_hull.push_back(points[curr_idx]);
    }

    convex_hull.pop_back();
    cout << convex_hull.size() << '\n';
    for (ii p : convex_hull) {
        cout << p.first << ' ' << p.second << '\n';
    }
    return 0;
}
