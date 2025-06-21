#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;
double coords[2][2];
int line_idx[2][2];

int get_lower_bound(vector<double> lines, double val) {
    int n = lines.size();
    if (n == 0) {
        return -1;
    }

    int l = 0, r = n - 1;
    while (l < r) {
        int mid = (l + r + 1) >> 1;
        if (lines[mid] > val) {
            r = mid - 1;
        } else {
            l = mid;
        }
    }
    if (lines[l] > val) {
        return -1;
    }
    if (lines[l] == val) {
        return l - 1;
    }
    return l;
}

int get_upper_bound(vector<double> lines, double val) {
    int n = lines.size();
    if (n == 0) {
        return -1;
    }

    int l = 0, r = n - 1;
    while (l < r) {
        int mid = (l + r) >> 1;
        if (lines[mid] > val) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    if (lines[r] <= val) {
        return -1;
    }
    return r;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int n;
    cin >> n;
    vector<double> lines[2][2];
    for (int i = 0; i < n; ++i) {
        int a, b, c;
        cin >> a >> b >> c;

        /*
        a == 0:
            b > 0 -> lines[1][1]
            b < 0 -> lines[1][0]
        
        b == 0:
            a > 0 -> lines[0][1]
            a < 0 -> lines[0][0]
        */

        if (a == 0) {
            lines[1][b > 0].push_back(-1.0 * c / b);
        } else {
            lines[0][a > 0].push_back(-1.0 * c / a);
        }
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            sort(lines[i][j].begin(), lines[i][j].end());
            // cout << lines[i][j].size() << '\n';
        }
    }

    int m;
    cin >> m;
    while (m--) {
        double x, y;
        cin >> x >> y;

        line_idx[0][0] = get_lower_bound(lines[0][0], x);
        line_idx[0][1] = get_upper_bound(lines[0][1], x);
        line_idx[1][0] = get_lower_bound(lines[1][0], y);
        line_idx[1][1] = get_upper_bound(lines[1][1], y);

        int min_num = 0;
        for (int i = 0; i < 4; ++i) {
            min_num = min(min_num, line_idx[(i >> 1) & 1][i & 1]);
        }

        if (min_num == -1) {
            cout << "NO\n";
            continue;
        }

        for (int i = 0; i < 4; ++i) {
            int idx = line_idx[(i >> 1) & 1][i & 1];
            coords[(i >> 1) & 1][i & 1] = lines[(i >> 1) & 1][i & 1][idx];
        }

        double dist_x = coords[0][1] - coords[0][0];
        double dist_y = coords[1][1] - coords[1][0];
        double area = dist_x * dist_y;
        cout << "YES\n" << fixed << setprecision(6) << area << '\n';
    }
    return 0;
}
