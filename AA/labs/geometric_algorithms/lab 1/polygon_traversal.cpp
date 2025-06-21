#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;

int get_orientation_value(ii p1, ii p2, ii p3) {
    int sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    int sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n, left_turns = 0, right_turns = 0, same_direction = 0;
    cin >> n;
    vector<ii> points(n);
    for (int i = 0; i < n; ++i) {
        cin >> points[i].first >> points[i].second;
    }
    points.push_back(points[0]);

    for (int i = 1; i < n; ++i) {
        int orientation = get_orientation_value(points[i - 1], points[i], points[i + 1]);
        left_turns += orientation > 0;
        right_turns += orientation < 0;
        same_direction += orientation == 0;
    }

    cout << left_turns <<  ' ' << right_turns << ' ' << same_direction << '\n';
    return 0;
}
