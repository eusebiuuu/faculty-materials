package main

import (
	"net/http"
	"fmt"
	"math/rand/v2"
	"strconv"
	"syscall"
	"context"
	"log"
	"os/signal"
	"time"
	"encoding/json"
	"io"
	"unicode"
	"math"
)

func random(writer http.ResponseWriter, req *http.Request) {
	fmt.Fprintf(writer, strconv.Itoa(rand.Int()))
}

func headers(w http.ResponseWriter, req *http.Request) {
	for name, headers := range req.Header {
        for _, h := range headers {
            fmt.Fprintf(w, "%v: %v\n", name, h)
        }
    }
}

type RequestPayload1 struct {
	Words  []string `json:"words"`
	Type int `json:"type"`
}

type ResponsePayload1 struct {
	Words  []string `json:"words"`
}

func exercise1(w http.ResponseWriter, req *http.Request) {
	fmt.Println("The server got the data")
	if req.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var payload RequestPayload1
	
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	words := payload.Words
	var count = len(words)
	var word_size = len(words[0])
	var new_words = make([]string, 0, word_size)

	for j := 0; j < word_size; j++ {
		var current_word = ""
		for i := 0; i < count; i++ {
			current_word += string(words[i][j])
		}
		new_words = append(new_words, current_word)
	}

	response := ResponsePayload1{
		Words: new_words,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	json.NewEncoder(w).Encode(response)
	fmt.Printf("The server is sending the response to the client: %s\n", new_words)
}

type RequestPayload2 struct {
	Words  []string `json:"words"`
}

type ResponsePayload2 struct {
	Result  string `json:"result"`
}

func isPerfectSquare(num int) (bool) {
	var root = math.Sqrt(float64(num))
	return int(root) * int(root) == num
}

func exercise2(w http.ResponseWriter, req *http.Request) {
	fmt.Println("The server got the data")
	if req.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var payload RequestPayload2
	
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	words := payload.Words
	var count = len(words)
	var final_answer = ""
	var perf_squares = 0

	for i := 0; i < count; i++ {
		var size = len(words[i])
		var curr_num = 0
		var seen_digit = false

		for j := 0; j < size; j++ {
			if unicode.IsDigit(rune(words[i][j])) {
				seen_digit = true
				var digit, _ = strconv.Atoi(string(words[i][j]))
				curr_num = curr_num * 10 + digit
			}
		}
		if seen_digit && isPerfectSquare(curr_num) {
			var answer = fmt.Sprintf("%d din %s, ", curr_num, words[i])
			final_answer += answer
			perf_squares++
		}
	}

	final_answer = fmt.Sprintf("%d perfect squares: %s", perf_squares, final_answer)
	response := ResponsePayload2{
		Result: final_answer,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	json.NewEncoder(w).Encode(response)
	fmt.Printf("The server is sending the response to the client: %s\n", final_answer)
}

type RequestPayload3 struct {
	Numbers  []int `json:"numbers"`
}

type ResponsePayload3 struct {
	Sum  int `json:"sum"`
}


func exercise3(w http.ResponseWriter, req *http.Request) {
	fmt.Println("The server got the data")
	if req.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var payload RequestPayload3
	
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	nums := payload.Numbers
	var count = len(nums)
	var sum = 0

	for i := 0; i < count; i++ {
		var temp_num = nums[i]
		var reverse_num = 0
		for j := 0; temp_num > 0; j++ {
			reverse_num = reverse_num * 10 + temp_num % 10
			temp_num /= 10
		}
		sum += reverse_num
	}

	response := ResponsePayload3{
		Sum: sum,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	json.NewEncoder(w).Encode(response)
	fmt.Printf("The server is sending the response to the client: %d\n", sum)
}

type RequestPayload4 struct {
	Numbers  []int `json:"numbers"`
	MinSum  int    `json:"minSum"`
	MaxSum  int    `json:"maxSum"`
}

type ResponsePayload4 struct {
	Average  float32 `json:"avg"`
}

func findSumOfDigits(num int) (int) {
	var sum = 0
	for i := 0; num > 0; i++ {
		sum += num % 10
		num /= 10
	}
	return sum
}

func exercise4(w http.ResponseWriter, req *http.Request) {
	fmt.Println("The server got the data")
	if req.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var payload RequestPayload4
	
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	nums := payload.Numbers
	var count = len(nums)
	var sum = 0
	var validCount = 0

	for i := 0; i < count; i++ {
		var currentSum = findSumOfDigits(nums[i])
		if payload.MinSum <= currentSum && currentSum <= payload.MaxSum {
			sum += nums[i]
			validCount++
		}
	}

	var avg float32 = float32(sum) / float32(validCount)
	response := ResponsePayload4{
		Average: avg,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	json.NewEncoder(w).Encode(response)
	fmt.Printf("The server is sending the response to the client: %f\n", avg)
}

type RequestPayload5 struct {
	Numbers  []int `json:"numbers"`
}

type ResponsePayload5 struct {
	Sum  int `json:"sum"`
}

func getDifference(num int) (int) {
	var temp_num = num
	var pw10 = 1
	var last_digit = 0

	for i := 0; temp_num > 0; i++ {
		last_digit = temp_num % 10
		temp_num /= 10
		pw10 *= 10
	}
	return num + last_digit * pw10
}

func exercise5(w http.ResponseWriter, req *http.Request) {
	fmt.Println("The server got the data")
	if req.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer req.Body.Close()

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var payload RequestPayload5
	
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "Invalid JSON format", http.StatusBadRequest)
		return
	}

	nums := payload.Numbers
	var count = len(nums)
	var sum = 0

	for i := 0; i < count; i++ {
		sum += getDifference(nums[i])
	}

	response := ResponsePayload5{
		Sum: sum,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	
	json.NewEncoder(w).Encode(response)
	fmt.Printf("The server is sending the response to the client: %d\n", sum)
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	server := &http.Server{
        Addr: ":8080",
    }

	http.HandleFunc("/random", random)
	http.HandleFunc("/headers", headers)
	http.HandleFunc("/1", exercise1)
	http.HandleFunc("/2", exercise2)
	http.HandleFunc("/3", exercise3)
	http.HandleFunc("/4", exercise4)
	http.HandleFunc("/5", exercise5)

	go func() {
		log.Println("Server starting on :8080...")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Could not listen and serve: %v", err)
		}
	}()

	<-ctx.Done()

	log.Println("Received shutdown signal, starting graceful shutdown...")

	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 3 * time.Second)
	defer cancelShutdown()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exiting gracefully.")
}