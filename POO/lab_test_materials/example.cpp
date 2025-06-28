#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>

class Stream {
public:
    Stream() = default;
    virtual ~Stream() = default;

    friend std::istream &operator>>(std::istream &in, Stream &ob) {
        ob.read(in);
        return in;
    }

    friend std::ostream &operator<<(std::ostream &out, const Stream &ob) {
        ob.write(out);
        return out;
    }

    virtual void read(std::istream &in){}
    virtual void write(std::ostream &out) const {}
};

template <class T>
class Checker {
public:
    Checker() = default;
    virtual ~Checker() = default;

    static void throwIfNotPositive(T);
    static void throwIfNotStrictPositive(T);
    static void throwIfNotInRange(T, T, T, const std::string& = "Numarul");
    static void throwIfGreaterThan(T, T, const std::string& = "Numarul");
};

template <class T>
void Checker<T>::throwIfNotPositive(T value) {
    if (value < 0) {
        throw std::runtime_error("Numarul trebuie sa fie pozitiv!");
    }
}

template <class T>
void Checker<T>::throwIfNotStrictPositive(T value) {
    if (value <= 0) {
        throw std::runtime_error("Numarul trebuie sa fie strict pozitiv!");
    }
}

template <class T>
void Checker<T>::throwIfNotInRange(T value, T a, T b, const std::string& ob_name) {
    if (value<a || value>b) {
        throw std::runtime_error(ob_name+" trebuie sa fie in intervalul ["+std::to_string(a)+", "+std::to_string(b)+"]");
    }
}

template <class T>
void Checker<T>::throwIfGreaterThan(T value, T a, const std::string& ob_name) {
    if (value>a) {
        throw std::runtime_error(ob_name+" trebuie sa fie < "+std::to_string(a));
    }
}

class Jucarie: public Stream {
protected:
    std::string denumire, tip;
    int dim = 0;
public:
    Jucarie() = default;
    Jucarie(std::string, const int &, std::string);

    void read(std::istream &in) override;

    void write(std::ostream &out) const override;

    ~Jucarie() override = default;
};

Jucarie::Jucarie(std::string den, const int &dim, std::string tip): denumire(std::move(den)), dim(dim), tip(std::move(tip)){}

void Jucarie::read(std::istream &in) {
    std::cout << "Denumire: \n";
    getline(in, denumire);
    getline(in, denumire);
    std::cout << "Dimensiune: \n";
    in >> dim;
    std::cout << "Tip: \n";
    getline(in, tip);
    getline(in, tip);
}

void Jucarie::write(std::ostream &out) const {
    out << "Denumire: " << denumire << '\n';
    out << "Dimensiune: " << dim << '\n';
    out << "Tip: " << tip << '\n';
}

class JucarieClasica: public Jucarie {
    std::string material, culoare;
public:
    JucarieClasica() = default;
    JucarieClasica(std::string, const int &, std::string, std::string, std::string);

    void read(std::istream &in) override;
    void write(std::ostream &out) const override;

    ~JucarieClasica() override = default;
};

JucarieClasica::JucarieClasica(std::string den, const int &dim, std::string tip, std::string mat, std::string cul): Jucarie(std::move(den), dim, std::move(tip)), material(std::move(mat)), culoare(std::move(cul)) {}

void JucarieClasica::read(std::istream &in) {
    Jucarie::read(in);
    std::cout << "Material: \n";
    getline(in, material);
    std::cout << "Culoare: \n";
    getline(in, culoare);
}

void JucarieClasica::write(std::ostream &out) const {
    Jucarie::write(out);
    out << "Material: " << material << '\n';
    out << "Culoare: " << culoare << '\n';
}

class JucarieElectronica: virtual public Jucarie {
protected:
    int num_bat = 0;
public:
    JucarieElectronica() = default;
    JucarieElectronica(std::string, const int &, std::string, int);

    void read(std::istream &in) override;
    void write(std::ostream &out) const override;

    ~JucarieElectronica() override = default;
};

JucarieElectronica::JucarieElectronica(std::string den, const int &dim, std::string tip, int num_bat): Jucarie(std::move(den), dim, std::move(tip)), num_bat(num_bat) {}

void JucarieElectronica::read(std::istream &in) {
    Jucarie::read(in);
    std::cout << "Numar baterii: \n";
    in >> num_bat;
}

void JucarieElectronica::write(std::ostream &out) const {
    Jucarie::write(out);
    out << "Numar baterii: " << num_bat << '\n';
}


class JucarieEducativa: virtual public Jucarie {
protected:
    std::string abilitate;
public:
    JucarieEducativa() = default;
    JucarieEducativa(std::string, const int &, std::string, std::string);

    void read(std::istream &in) override;

    void write(std::ostream &out) const override;
};

JucarieEducativa::JucarieEducativa(std::string den, const int &dim, std::string tip, std::string abilitate): Jucarie(std::move(den), dim, std::move(tip)), abilitate(std::move(abilitate)) {}

void JucarieEducativa::read(std::istream &in) {
    Jucarie::read(in);
    std::cout << "Abilitate: \n";
    getline(in, abilitate);
}

void JucarieEducativa::write(std::ostream &out) const {
    Jucarie::write(out);
    out << "Abilitate: " << abilitate << '\n';
}


class JucarieModerna: public JucarieEducativa, public JucarieElectronica {
    std::string brand, model;
public:
    JucarieModerna() = default;
    JucarieModerna(const std::string& den, const int &dim, const std::string& tip, const std::string& brand, const std::string& model): Jucarie(den, dim, tip),
                                                                                                            JucarieEducativa(den, dim, tip, "generala"),
                                                                                                            JucarieElectronica(den, dim, tip, 1), brand(brand), model(model) {}

    void read(std::istream &in) override;
    void write(std::ostream &out) const override;

    ~JucarieModerna() override = default;
};

void JucarieModerna::read(std::istream &in) {
    Jucarie::read(in);
    abilitate = "generala";
    num_bat = 1;
    std::cout << "Brand: \n";
    getline(in, brand);
    std::cout << "Model: \n";
    getline(in, model);
}

void JucarieModerna::write(std::ostream &out) const {
    JucarieEducativa::write(out);
    out << "Numar baterii: " << num_bat << '\n';
    out << "Brand: " << brand << '\n';
    out << "Model: " << model << '\n';
}

class JucarieFactory {
    JucarieFactory() = default;
public:
    static Jucarie *copyInstance(Jucarie *jucarie) {
        if (auto var1 = dynamic_cast<JucarieClasica *>(jucarie))
            return new JucarieClasica(*var1);
        else if (auto var2 = dynamic_cast<JucarieEducativa *>(jucarie))
            return new JucarieEducativa(*var2);
        else if (auto var3 = dynamic_cast<JucarieElectronica *>(jucarie))
            return new JucarieElectronica(*var3);
        else if (auto var4 = dynamic_cast<JucarieModerna *>(jucarie))
            return new JucarieModerna(*var4);
        else if (jucarie == nullptr)
            return nullptr;
        throw std::runtime_error("Tipul jucariei este invalid!");
    }

    static Jucarie *newInstance(const std::string &tip) {
        if (tip == "clasica") {
            return new JucarieClasica();
        } else if (tip == "educativa") {
            return new JucarieEducativa();
        } else if (tip == "electronica") {
            return new JucarieElectronica();
        } else if (tip == "moderna") {
            return new JucarieModerna();
        } else {
            throw std::runtime_error("Tip de jucarie invalid");
        }
    }
};

class Copil: public Stream {
protected:
    static int count;
    int ID;
    std::string nume, prenume, adresa;
    int varsta, fapteBune;
    std::vector<Jucarie *> jucarii;
public:
    Copil() {
        ID = Copil::count++;
        varsta = fapteBune = 0;
    }
    Copil(const std::string &, const std::string &, const std::string &, int, int, const std::vector<Jucarie *> &);

    void read(std::istream &in) override;

    void write(std::ostream &out) const override;

    bool operator<(const Copil &copil) const {
        return this->varsta < copil.varsta;
    }

    [[nodiscard]] int getFapteBune() const;

    Copil *operator+=(int x) {
        this->fapteBune += x;
        for (int i = 0; i < x; ++i) {
            std::cout << "Tip (clasica / educativa / moderna / electronica): \n";
            std::string tip;
            std::cin >> tip;
            Jucarie *j = JucarieFactory::newInstance(tip);
            std::cin >> *j;
            jucarii.push_back(j);
        }
        return this;
    }

    [[nodiscard]] int getId() const;

    Copil(const Copil &c) {
        this->nume = c.nume;
        this->prenume = c.prenume;
        this->adresa = c.adresa;
        this->fapteBune = c.fapteBune;
        this->varsta = c.varsta;
        this->ID = c.ID;
        jucarii.clear();
        for (auto *j : c.jucarii) {
            jucarii.push_back(JucarieFactory::copyInstance(j));
        }
    }

    ~Copil() override {
        for (auto *j : jucarii) {
            delete j;
        }
        jucarii.clear();
    }

    [[nodiscard]] const std::string &getNume() const;
};

int Copil::count = 0;

Copil::Copil(const std::string &nume, const std::string &prenume, const std::string &adresa, int v, int fb, const std::vector<Jucarie *> &jj): nume(nume), prenume(prenume), adresa(adresa), varsta(v), fapteBune(fb), jucarii(jj) {
    ID = Copil::count++;
}

void Copil::read(std::istream &in) {
    std::cout << "Nume: \n";
    getline(in, nume);
    getline(in, nume);
    std::cout << "Prenume: \n";
    getline(in, prenume);
    std::cout << "Adresa: \n";
    getline(in, adresa);
    std::cout << "Varsta: \n";
    in >> varsta;
    std::cout << "Fapte bune: \n";
    in >> fapteBune;
    for (int i = 0; i < fapteBune; ++i) {
        std::cout << "Tip (clasica / educativa / moderna / electronica): \n";
        std::string tip;
        in >> tip;
        Jucarie *j = JucarieFactory::newInstance(tip);
        in >> *j;
        jucarii.push_back(j);
    }
}

void Copil::write(std::ostream &out) const {
    out << "Nume: " << nume << '\n';
    out << "Prenume: " << prenume << '\n';
    out << "Adresa: " << adresa << '\n';
    out << "Varsta: " << varsta << '\n';
    out << "Fapte bune: " << fapteBune << '\n';
    out << "Jucarii: ";
    for (int i = 0; i < fapteBune; ++i) {
        if (jucarii.size() == i) {
            throw std::runtime_error("Nu sunt destule jucarii");
        }
        out << *jucarii[i];
    }
}

int Copil::getFapteBune() const {
    return fapteBune;
}

int Copil::getId() const {
    return ID;
}

const std::string &Copil::getNume() const {
    return nume;
}

class CopilCuminte: public Copil {
    int num_dul;
public:
    CopilCuminte() {
        num_dul = 0;
    }
    CopilCuminte(const std::string &, const std::string &, const std::string &, int, int, const std::vector<Jucarie *> &, int);

    void read(std::istream &in) override;
    void write(std::ostream &out) const override;
};

CopilCuminte::CopilCuminte(const std::string &nume, const std::string &prenume, const std::string &adresa, int v, int fb, const std::vector<Jucarie *> &jj, int num_dul): Copil(nume, prenume, adresa, v, fb, jj), num_dul(num_dul) {}

void CopilCuminte::read(std::istream &in) {
    Copil::read(in);
    std::cout << "Numar dulciuri: \n";
    in >> num_dul;
}

void CopilCuminte::write(std::ostream &out) const {
    Copil::write(out);
    out << "Numar dulciuri: " << num_dul << '\n';
}


class CopilNeastamparat: public Copil {
    int carbuni;
public:
    CopilNeastamparat() {
        carbuni = 0;
    }
    CopilNeastamparat(const std::string &, const std::string &, const std::string &, int, int, const std::vector<Jucarie *> &, int);

    void read(std::istream &in) override;
    void write(std::ostream &out) const override;
};

CopilNeastamparat::CopilNeastamparat(const std::string &nume, const std::string &prenume, const std::string &adresa, int v, int fb, const std::vector<Jucarie *> &jj, int car): Copil(nume, prenume, adresa, v, fb, jj), carbuni(car) {}

void CopilNeastamparat::read(std::istream &in) {
    Copil::read(in);
    std::cout << "Numar carbuni: \n";
    in >> carbuni;
}

void CopilNeastamparat::write(std::ostream &out) const {
    Copil::write(out);
    out << "Numar carbuni: " << carbuni << '\n';
}

class CopilFactory {
    CopilFactory() = default;
public:
    static Copil *copyInstance(Copil *copil) {
        if (auto c1 = dynamic_cast<CopilCuminte *>(copil)) {
            return new CopilCuminte(*c1);
        } else if (auto c2 = dynamic_cast<CopilNeastamparat *>(copil)) {
            return new CopilNeastamparat(*c2);
        } else {
            throw std::runtime_error("Tip de copil invalid");
        }
    }

    static Copil *getInstance(const std::string &tip) {
        if (tip == "good") {
            auto *c = new CopilCuminte();
            std::cin >> *c;
            return c;
        } else if (tip == "bad") {
            auto *c = new CopilNeastamparat();
            std::cin >> *c;
            return c;
        } else {
            throw std::runtime_error("Tip de copil invalid");
        }
    }
};

class Lume {
    Lume() = default;

    inline static Lume *lume;

    ~Lume() {
        for (auto *c : copii) {
            delete c;
        }
        for (auto *j : jucariiCreate) {
            delete j;
        }
        copii.clear();
        jucariiCreate.clear();
    }

    std::vector<Copil *> copii;
    std::vector<Jucarie *> jucariiCreate;
public:

    static Lume *getInstance() {
        if (lume == nullptr)
            lume = new Lume();
        return lume;
    }

    void citireNCopii();

    void afisareNCopii();

    void gasesteCopil();

    void ordonareCopiiVarsta();

    void ordonareCopiiFapteBune();

    void adaugareInCopil();

    void afisareRaport();
};

void Lume::citireNCopii() {
    std::cout << "Numar copii: \n";
    int n;
    std::cin >> n;
    for (int i = 0; i < n; ++i) {
        std::cout << "Tip copil (good / bad): \n";
        std::string tip;
        std::cin >> tip;
        this->copii.push_back(CopilFactory::getInstance(tip));
    }
}

void Lume::afisareNCopii() {
    for (auto *c : copii) {
        std::cout << *c;
    }
}

void Lume::ordonareCopiiVarsta() {
    std::sort(copii.begin(), copii.end());
}

void Lume::ordonareCopiiFapteBune() {
    std::sort(copii.begin(), copii.end(), [](Copil *c1, Copil *c2) {
        return c1->getFapteBune() > c2->getFapteBune();
    });
}

void Lume::adaugareInCopil() {
    int id;
    std::cout << "Id copil: \n";
    std::cin >> id;
    bool found = false;
    for (auto *c : copii) {
        if (c->getId() == id) {
            found = true;
            std::cout << "Introdu nr de fapte bune de adaugat: \n";
            int x;
            std::cin >> x;
            *c += x;
        }
    }
    if (!found) {
        throw std::runtime_error("Niciun copil cu id-ul introdus");
    }
}

void Lume::gasesteCopil() {
    std::string nume;
    std::cout << "Nume copil: ";
    getline(std::cin, nume);
    getline(std::cin, nume);
    std::cout << "Copii cu nume similar: \n";
    for (auto *copil : copii) {
        if (copil->getNume().find(nume) != std::string::npos) {
            std::cout << *copil;
        }
    }
}

void Lume::afisareRaport() {
    std::cout << "Raport\n";
}

void printMenu() {
    std::cout << "1: Citire n copii (+ jucarii)\n";
    std::cout << "2: Afisare n copii (+ jucarii) \n";
    std::cout << "3: Gaseste copil dupa nume\n";
    std::cout << "4: Ordonare copii dupa varsta\n";
    std::cout << "5: Ordonare copii dupa fapte bune\n";
    std::cout << "6: Adaugare fapte bune si jucarii la un copil\n";
    std::cout << "7: Afisare raport\n";
    std::cout << "8: Exit\n";
}

void menu() {
    int command;
    do {
        printMenu();
        std::cin >> command;
//        std::cout << command << '\n';
        try {
            Checker<int>::throwIfNotInRange(command, 1, 8, "Comanda");
        } catch (const std::exception &ex) {
            std::cout << ex.what() << '\n';
        }
        switch(command) {
            case 1:
                try {
                    Lume::getInstance()->citireNCopii();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 2:
                try {
                    Lume::getInstance()->afisareNCopii();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 3:
                try {
                    Lume::getInstance()->gasesteCopil();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 4:
                try {
                    Lume::getInstance()->ordonareCopiiVarsta();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 5:
                try {
                    Lume::getInstance()->ordonareCopiiFapteBune();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 6:
                try {
                    Lume::getInstance()->adaugareInCopil();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            case 7:
                try {
                    Lume::getInstance()->afisareRaport();
                } catch (const std::exception &ex) {
                    std::cout << ex.what() << '\n';
                }
                break;
            default:
                return;
        }
    } while (true);
}

int main() {
    menu();
    return 0;
}