package main

import (
	"bytes"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"time"
	"os"
)

type RequestPayload5 struct {
	Numbers  []int `json:"numbers"`
}

type ResponsePayload5 struct {
	Sum  int `json:"sum"`
}

func exercise() (RequestPayload5, string, error) {
	filename := "input.txt"

	file, err := os.Open(filename)

	if err != nil {
		return RequestPayload5{}, "", fmt.Errorf("error opening file %s: %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var count int

	scanner.Scan()
	firstLine := scanner.Text()
	count, err = strconv.Atoi(firstLine)

	url := "http://localhost:8080/5"

	var numbers = make([]int, 0, count)
	for i := 0; i < count; i++ {
		scanner.Scan()
		var num, _ = strconv.Atoi(scanner.Text())
		numbers = append(numbers, num)
	}

	data := RequestPayload5{
		Numbers: numbers,
	}

	return data, url, nil
}

func main() {
	fmt.Println("The client 5 has connected")
	var bodyReader *bytes.Buffer

	data, url, err := exercise()
	jsonPayload, err := json.Marshal(data)
	if err != nil {
		log.Fatalf("Error marshaling payload: %v", err)
	}
	bodyReader = bytes.NewBuffer(jsonPayload)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		log.Fatalf("Error creating request: %v", err)
	}
	
	req.Header.Set("Content-Type", "application/json")
	
	fmt.Printf("The client is making the following request: %s\n", string(jsonPayload))
	resp, err := http.DefaultClient.Do(req)

	if err != nil {
		log.Fatalf("Error sending request: %v", err)
	}
	defer resp.Body.Close()

	// Read and print the server's response
	responseBody, _ := io.ReadAll(resp.Body)
	// fmt.Printf("Server Status: %s\n", resp.Status)
	// fmt.Printf("Server Response Body: %s\n", string(responseBody))
	fmt.Printf("The client got the response from the server: %s\n", string(responseBody))
}