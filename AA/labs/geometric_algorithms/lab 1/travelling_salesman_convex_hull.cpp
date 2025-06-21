#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;
int const P = 1004;
/*
- Sort by x & find indexes for hull points in std::vector
- Convex hull & store in std::list & bitset of seen indexes
- Find closest sides to inner points & keep order from hull
- Data structures: 1 hashmap <ii, vector<int>> & 1 multiset pair<ii, int>
- TSP
    - Extract minimum value from multiset
    - Get all the points of the side (except the current one)
    - Insert the current point in the hull
    - For all idle points find the best from the 2 new sides & store in multiset
*/

ll get_orientation_value(ii p1, ii p2, ii p3) {
    ll sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    ll sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

long double get_dist(ii p1, ii p2) {
    return sqrt((p1.first - p2.first) * (p1.first - p2.first) + (p1.second - p2.second) * (p1.second - p2.second));
}

long double get_cost(ii p1, ii p2, ii p3) {
    return get_dist(p1, p3) + get_dist(p2, p3) - get_dist(p1, p2);
}

long double get_storing_cost(ii p1, ii p2, ii p3) {
    long double c1 = get_dist(p1, p3);
    long double c2 = get_dist(p2, p3);
    long double c3 = get_dist(p1, p2);
    return (c1 + c2) / c3;
}

ii get_minimal_side(list<int> &tsp_hull, int point_idx, vector<ii> &points) {
    auto it = tsp_hull.begin(); it++;

    long double min_cost = 1e9;
    ii side = {0, 0};

    while (it != tsp_hull.end()) {
        auto prev_it = prev(it, 1);
        long double curr_cost = get_cost(points[*prev_it], points[*it], points[point_idx]);
        if (min_cost > curr_cost) {
            min_cost = curr_cost;
            side = {*prev_it, *it};
        }
        it++;
    }

    return side;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n, min_x = 1e9, min_idx = 0;
    cin >> n;
    vector<ii> point(n);
    list<int> tsp_hull;
    for (int i = 0; i < n; ++i) {
        cin >> point[i].first >> point[i].second;
        if (point[i].first < min_x) {
            min_idx = i;
            min_x = point[i].first;
        }
    }
    swap(point[0], point[min_idx]);
    ii p0 = point[0];

    sort(point.begin() + 1, point.end(), [&p0](const ii& a, const ii& b) {
        int o = get_orientation_value(p0, a, b);
        if (o == 0)
            return (p0.first-a.first)*(p0.first-a.first) + (p0.second-a.second)*(p0.second-a.second)
                < (p0.first-b.first)*(p0.first-b.first) + (p0.second-b.second)*(p0.second-b.second);
        return o > 0;
    });

    tsp_hull.push_back(0);
    tsp_hull.push_back(1);

    for (int i = 2; i <= n; ++i) {
        int idx = i % n;

        while (tsp_hull.size() > 1) {
            auto itr = prev(tsp_hull.end(), 2);
            ll orientation = get_orientation_value(point[*itr], point[tsp_hull.back()], point[idx]);
            if (orientation > 0) {
                break;
            }
            tsp_hull.pop_back();
        }

        tsp_hull.push_back(idx);
    }

    bitset<P> seen;
    for (int p : tsp_hull) {
        seen[p] = true;
    }

    map<ii, vector<int>> points;
    multiset<pair<pair<long double, int>, ii>> min_dist;

    for (int i = 0; i < n; ++i) {
        if (seen[i]) {
            continue;
        }

        ii best_side = get_minimal_side(tsp_hull, i, point);
        min_dist.insert({{get_storing_cost(point[best_side.first], point[best_side.second], point[i]), i}, best_side});
        points[best_side].push_back(i);
    }

    while (!min_dist.empty()) {
        auto it = min_dist.begin();
        ii side = (*it).second;
        int idx = (*it).first.second;
        min_dist.erase(it);

        if (seen[idx]) {
            continue;
        }

        vector<int> idle_points;
        for (int p : points[side]) {
            if (p != idx) {
                idle_points.push_back(p);
            }
        }

        auto insert_it = tsp_hull.begin(); insert_it++;
        while (insert_it != tsp_hull.end()) {
            if (*insert_it == side.second) {
                break;
            }
            insert_it++;
        }

        tsp_hull.insert(insert_it, idx);
        seen[idx] = true;
        int p3 = *insert_it, p2 = *prev(insert_it, 1), p1 = *prev(insert_it, 2);
        
        for (int pos : idle_points) {
            long double c1 = get_cost(point[p1], point[p2], point[pos]);
            long double c2 = get_cost(point[p2], point[p3], point[pos]);
            ii curr_side = c1 <= c2 ? make_pair(p1, p2) : make_pair(p2, p3);

            points[curr_side].push_back(pos);
            min_dist.insert({{get_storing_cost(point[curr_side.first], point[curr_side.second], point[pos]), pos}, curr_side});
        }
    }

    for (int p : tsp_hull) {
        cout << point[p].first << ' ' << point[p].second << '\n';
    }
    return 0;
}
