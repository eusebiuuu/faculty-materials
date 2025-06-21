#include <bits/stdc++.h>
#define ll long long
#define ii pair<ll, ll>
using namespace std;
int const INF = 1e9 + 9;

ll get_determinant_value(ii p1, ii p2, ii p3) {
    ll sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    ll sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

string get_point_place(ii query_point, vector<ii> &points, int n, bool print_flag) {
    int l = 1, r = n - 2;
    while (l < r) {
        int mid = (l + r + 1) / 2;
        ll orientation = get_determinant_value(points[0], query_point, points[mid]);
        if (orientation > 0) {
            r = mid - 1;
        } else {
            l = mid;
        }
    }

    // cout << l << '\n';

    ll target_area = abs(get_determinant_value(points[0], points[l], points[l + 1]));
    ll triangle_1 = abs(get_determinant_value(query_point, points[0], points[l]));
    ll triangle_2 = abs(get_determinant_value(query_point, points[0], points[l + 1]));
    ll triangle_3 = abs(get_determinant_value(query_point, points[l], points[l + 1]));

    // cout << triangle_1 << ' ' << triangle_2 << ' ' << triangle_3 << ' ' << target_area << '\n';

    if (target_area == triangle_1 + triangle_2 + triangle_3) {
        ll turn = abs(get_determinant_value(points[l], points[l + 1], query_point));
        if (l == n - 2) {
            turn = min(turn, abs(get_determinant_value(points[0], points[n - 1], query_point)));
        }
        turn = min(turn, abs(get_determinant_value(points[l - 1], points[l], query_point)));

        if (print_flag) {
            cout << turn << '\n';
            cout << query_point.first << ' ' << query_point.second << '\n';
            cout << l << '\n';
            cout << points[l].first << ' ' << points[l].second << '\n';
            cout << points[l + 1].first << ' ' << points[l + 1].second << '\n';
            cout << points[0].first << ' ' << points[0].second << '\n';
        }

        if (turn == 0) {
            return "BOUNDARY";
        }

        return "INSIDE";
    }
    return "OUTSIDE";
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n, start = 0, y_coord = -INF, min_x = INF;
    cin >> n;
    vector<ii> points(n);
    for (int i = 0; i < n; ++i) {
        cin >> points[i].first >> points[i].second;
        if (points[i].first <= min_x) {
            if (points[i].first == min_x) {
                if (y_coord < points[i].second) {
                    y_coord = points[i].second;
                    start = i;
                }
            } else {
                min_x = points[i].first;
                y_coord = points[i].second;
                start = i;
            }
        }
    }

    rotate(points.begin(), points.begin() + start, points.end());
    
    int queries, c = 0;
    cin >> queries;
    while (queries--) {
        c++;
        ii p;
        cin >> p.first >> p.second;
        cout << get_point_place(p, points, n, false) << '\n';
    }
    return 0;
}
