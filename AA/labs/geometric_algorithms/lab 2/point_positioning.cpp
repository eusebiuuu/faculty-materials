#include <bits/stdc++.h>
#define ll long long
#define ii pair<ll, ll>
using namespace std;
ll const INF = 1e9 + 9;
ii const P1 = {INF, INF}, P2 = {INF, -INF};

ll get_orientation_value(ii p1, ii p2, ii p3) {
    ll sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    ll sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

bool intersect(ll val1, ll val2) {
    return (val1 < 0 && val2 > 0) || (val2 < 0 && val1 > 0) || ((val1 == 0) ^ (val2 == 0));
}

bool strict_intersect(ll val1, ll val2) {
    return (val1 < 0 && val2 > 0) || (val2 < 0 && val1 > 0);
}

bool segments_intersection(ii p1, ii p2, ii p3, ii p4, int type = 2) {
    ll orientation_11 = get_orientation_value(p1, p2, p3);
    ll orientation_12 = get_orientation_value(p1, p2, p4);
    ll orientation_21 = get_orientation_value(p3, p4, p1);
    ll orientation_22 = get_orientation_value(p3, p4, p2);
    
    if (type == 0) { // for vertice intersection
        return strict_intersect(orientation_11, orientation_12);
    } else if (type == 1) { // for segment intersection
        return strict_intersect(orientation_11, orientation_12) && strict_intersect(orientation_21, orientation_22);
    }
    return intersect(orientation_11, orientation_12) && intersect(orientation_21, orientation_22);
}

bool is_point_in_segment(ii p1, ii p2, ii p3) {
    return get_orientation_value(p1, p2, p3) == 0 && (
        segments_intersection(p1, p2, P1, p3) || segments_intersection(p1, p2, P2, p3)
    );
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    vector<ii> points(n);
    for (ii &elem : points) {
        cin >> elem.first >> elem.second;
    }
    int m;
    cin >> m;
    while (m--) {
        ii query_point;
        cin >> query_point.first >> query_point.second;

        bool boundary = false;
        int intersection_count = 0;

        for (int i = 0; i < n && !boundary; ++i) {
            int j = (i + 1) % n;
            int k = (i + 2) % n;
            boundary |= is_point_in_segment(points[i], points[j], query_point);

            if (is_point_in_segment(P1, query_point, points[j])) {
                intersection_count ^= segments_intersection(P1, query_point, points[i], points[k], 0);
            } else {
                intersection_count ^= segments_intersection(points[i], points[j], query_point, P1, 1);
            }
        }

        if (boundary) {
            cout << "BOUNDARY\n";
            continue;
        }

        cout << (intersection_count == 1 ? "INSIDE" : "OUTSIDE") << '\n';
    }
    return 0;
}
