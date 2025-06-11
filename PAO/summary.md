- Dacă o clasă serializabilă extinde o clasă neserializabilă, atunci datele membre accesibile ale superclasei  nu vor fi serializate. În acest caz, superclasa trebuie să conțină un constructor fără argumente pentru a  inițializa în procesul de restaurare a obiectului datele membre moștenite.

---

Dacă un obiect care trebuie serializat conține referințe către obiecte neserializabile, atunci va fi generată  o excepție de tipul NotSerializableException.

---

Dacă a doua regulă nu este respectată, adică două obiecte egale din punct de vedere al conținutului (metoda 
equals) au hash-code-uri diferite (metoda hashCode), atunci operațiile de căutare/inserare într-o tabelă de 
dispersie vor fi incorecte. Astfel, în cazul în care se încearcă inserarea celui de-al doilea obiect după inserarea 
primului, operația de căutare a celui de-al doilea obiect se va efectua după valoarea hash-code-ului său, diferită 
de cea a primului obiect, deci îl va căuta în alt bucket și nu îl va găsi, ceea ce va conduce la inserarea și a celui 
de-al doilea obiect în tabela, deși el are același conținut cu primul obiect! 
De obicei, acest aspect negativ apare în momentul în care programatorul nu rescrie metodele hashCode și 
equals într-o clasă ale cărei instanțe vor fi utilizate în cadrul unor colecții bazate pe tabele de dispersie, 
deoarece, implicit, metoda hashCode furnizează o valoare calculată pe baza referinței obiectului respectiv, iar 
metoda equals testează egalitatea a două obiecte comparând referințele lor. Astfel, două obiecte diferite cu 
același conținut vor fi considerate diferite de metoda equals și vor avea hash-code-uri diferite! 

---

Predicate -> test
Consumer -> accept
Function -> apply
Supplier -> get


---

Deschidere: from list, Stream.of, generate

Intermediare: filter, sorted(Comparator), sorted, limit, distinct, map, flatMap

Inchidere: forEach, min, max, collect(Collector)

Collectors: toList, toSet, toMap, groupingBy, joining, counting, averagingDouble, summingDouble


import java.sql.*;
import java.util.Scanner;

public class Main {
    private final static Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        double s = scanner.nextDouble();
        int v = scanner.nextInt();

        try {
            Connection connection = DriverManager.getConnection("jdbc:derby://localhost:1527/Angajati");
            Statement statement = connection.createStatement();
            ResultSet resultSet = statement.executeQuery(
                "SELECT *" + 
                "\nFROM DateAngajati" +
                "\nWHERE Varsta <= " + v + " AND Salariu >= " + s);

            while (resultSet.next()) {
                System.out.println("CNP: " + resultSet.getString("CNP"));
                System.out.println("Nume: " + resultSet.getString("Nume"));
                System.out.println("Varsta: " + resultSet.getInt("Varsta"));
                System.out.println("Salariu: " + resultSet.getDouble("Salariu"));
                System.out.println();
            }

            connection.close();
        } 
        catch (SQLException e) {
            e.printStackTrace();
        }
    }
}



import java.awt.*;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.averagingDouble;

//class A {
//    int x = 10;
//    static int y = 20;
//
//    int getX() {
//        return x;
//    }
//}
//
//class B extends A {
//    int x = 30;
//    static int y = 40;
//
//    int getX() {
//        return x;
//    }
//}


//public class Main {
//    static String sir = "A";
//
//    void A() {
//        try {
//            sir = sir + "B";
//            B();
//        } catch (Exception e) {
//            sir = sir + "C";
//        }
//    }
//
//    void B() throws Exception {
//        try {
//            sir = sir + "D";
//            C();
//        } catch (Exception e) {
//            throw new Exception();
//        } finally {
//            sir = sir + "E";
//        }
//    }
//
//    void C() throws Exception {
//        throw new Exception();
//    }
//
//    public static void main(String[] args) {
//        Main ob = new Main();
//        ob.A();
//        System.out.println(sir);
//    }
//}
//
//import java.util.*;

//public class Main {
//    public static void main(String[] args) {
//        List<Integer> numere = new ArrayList<Integer>();
//
//        for (int i = 0; i < 11; i++)
//            numere.add(i);
//
//        Iterator<Integer> itr = numere.iterator();
//        while (itr.hasNext()) {
//            Integer nr = itr.next();
//            if (nr % 2 == 0)
//                numere.remove(nr); // ⚠ This line will cause ConcurrentModificationException
//        }
//
//        System.out.println(numere);
//    }
//}

//class Adresa {
//    private String strada;
//    private String bloc;
//
//    public Adresa(String strada, String bloc) {
//        this.strada = strada;
//        this.bloc = bloc;
//    }
//
//    public String getStrada() {
//        return strada;
//    }
//
//    public void setStrada(String strada) {
//        this.strada = strada;
//    }
//
//    public String getBloc() {
//        return bloc;
//    }
//
//    public void setBloc(String bloc) {
//        this.bloc = bloc;
//    }
//}
//
//final class Facultate {
//    private final String name;
//    private final int students_count;
//    private final Adresa adress;
//
//    private final ArrayList<String> specializari;
//
//    public Facultate(String n, int cnt, Adresa adrs, ArrayList<String> spec) {
//        this.name = n;
//        this.students_count = cnt;
//        this.specializari = new ArrayList<>(spec);
//        this.adress = new Adresa(adrs.getStrada(), adrs.getBloc());
//    }
//
//
//    public String getName() {
//        return name;
//    }
//
//    public int getStudents_count() {
//        return students_count;
//    }
//
//    public Adresa getAdress() {
//        return new Adresa(this.adress.getStrada(), this.adress.getBloc());
//    }
//}

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

//class TotalValue {
//    private static Float total_value = 0.0f;
//
//    synchronized public static void increaseValue(Float val) {
//        total_value += val;
//    }
//
//    public static Float getTotal_value() {
//        return total_value;
//    }
//}
//
//class FirExecutie extends Thread {
//    private String filename;
//    private String firma;
//
//    public FirExecutie(String f, String firma) {
//        filename = f;
//        this.firma = firma;
//    }
//    @Override
//    public void run() {
//        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
//            String line;
//            while ((line = reader.readLine()) != null) {
//                String[] values = line.split(",");
//                if (values.length != 4) {
//                    throw new IllegalArgumentException("Liniile nu respecta formatul");
//                }
//                if (values[0].equals(firma)) {
//                    Float cantitate = Float.parseFloat(values[2]);
//                    Float pret_unitar = Float.parseFloat(values[3]);
//                    TotalValue.increaseValue(cantitate * pret_unitar);
//                }
//            }
//        } catch (Exception e) {
//            System.out.println(e.getMessage());
//        }
//    }
//}

interface Functie {
    Function<Integer, Double> f = x -> 1.0 / x;
}

class A implements Functie {
    public double f(int x) {  // This is NOT overriding the interface's field!
        return 2.0 / x;
    }
}

class Main {
    static void afisare(Functie f, int t) {
        System.out.println(f.f.apply(t));  // Calls the interface's lambda
         System.out.println(f.f(t));     // ERROR: 'f' is not a method in Functie
    }

    public static void main(String[] args) {
        afisare(null, 10);                          // NullPointerException
        afisare(new A().f, 20);                       // Uses interface's lambda (output: 0.05)
        afisare(null, (int) new A().f(1));          // NullPointerException
        afisare(new Functie() {}, 10);              // Uses interface's lambda (output: 0.1)
        afisare(x -> 3.0 / x, 20);                  // Error: Lambda doesn't implement Functie
    }
}

//public class Main {
//    public static void main(String[] args) {
//        Produs p1 = new Produs("continental", "masina", 50.0f, 70.0f);
//        Produs p2 = new Produs("BMW", "cauciuc", 80.0f, 100.0f);
//        Produs p3 = new Produs("Facebook", "software", 100f, 500f);
//        Produs p4 = new Produs("Facebook", "hardware", 500f, 1000f);
//
//        ArrayList<Produs> products = new ArrayList<Produs>();
//        products.add(p1);
//        products.add(p2);
//        products.add(p3);
//        products.add(p4);
//
//        products.stream().filter(p -> p.getPret_unitar() >= 100)
//                .sorted(Comparator.comparing(Produs::getCantitate).reversed()).forEach(System.out::println);
//
//        Comparator<String> comp = (s1, s2) -> s1.compareTo(s2);
//        products.stream().map(Produs::getFirma).distinct().sorted(comp).forEach(System.out::println);
//
//        List<Produs> filteredProducts = products.stream().filter(p -> p.getCantitate() * p.getPret_unitar() <= 10000).toList();
//        filteredProducts.forEach(System.out::println);
//
//        System.out.println(products.stream().map(Produs::getPret_unitar).collect(averagingDouble(pret -> pret)).toString());
//
//        Map<String, List<Produs>> produse = products.stream().collect(Collectors.groupingBy(Produs::getFirma));
//        System.out.println(produse);
//        Scanner scanner = new Scanner(System.in);
//
//        String firma = scanner.next();
//        FirExecutie f1 = new FirExecutie("src/magazin_1.txt", firma);
//        FirExecutie f2 = new FirExecutie("src/magazin_2.txt", firma);
//
//        f1.start();
//        f2.start();
//
//        try {
//            f1.join();
//            f2.join();
//        } catch (InterruptedException e) {
//            System.out.println(Arrays.toString(e.getStackTrace()));
//        }
//        System.out.println(TotalValue.getTotal_value());

//        LinkedHashMap m = new LinkedHashMap();
//        m.put('a', null);
//        m.put('b', "JavaSE");
//        m.put('c', "JSE");
//        m.put('c', "Python");
//        m.put(null, "PHP");
//        m.put(null, null);
//
//        System.out.println(m);
//    }
//}