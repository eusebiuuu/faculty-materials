#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    int tests;
    cin >> tests;
    while (tests--) {
        ll xp, yp, xq, yq, xr, yr;
        cin >> xp >> yp >> xq >> yq >> xr >> yr;
        ll orientation_value = xp * yq + xq * yr + xr * yp - (xr * yq + xq * yp + xp * yr);
        if (orientation_value > 0) {
            cout << "LEFT\n";
        } else if (orientation_value < 0) {
            cout << "RIGHT\n";
        } else {
            cout << "TOUCH\n";
        }
    }
    return 0;
}
