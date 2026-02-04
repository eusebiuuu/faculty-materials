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

type RequestPayload4 struct {
	Numbers  []int `json:"numbers"`
	MinSum  int    `json:"minSum"`
	MaxSum  int    `json:"maxSum"`
}

type ResponsePayload4 struct {
	Average  float32 `json:"avg"`
}

func exercise() (RequestPayload4, string, error) {
	filename := "/home/eusebiuu/Documents/software_projects/distributed_systems/homework_1/client 4/input.txt"

	file, err := os.Open(filename)

	if err != nil {
		return RequestPayload4{}, "", fmt.Errorf("error opening file %s: %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var count int
	var minSum int
	var maxSum int

	scanner.Scan()
	firstLine := scanner.Text()
	minSum, err = strconv.Atoi(firstLine)

	scanner.Scan()
	firstLine = scanner.Text()
	maxSum, err = strconv.Atoi(firstLine)

	scanner.Scan()
	firstLine = scanner.Text()
	count, err = strconv.Atoi(firstLine)

	url := "http://localhost:8080/4"

	var numbers = make([]int, 0, count)
	for i := 0; i < count; i++ {
		scanner.Scan()
		var num, _ = strconv.Atoi(scanner.Text())
		numbers = append(numbers, num)
	}

	data := RequestPayload4{
		Numbers: numbers,
		MinSum: minSum,
		MaxSum: maxSum,
	}

	return data, url, nil
}

func main() {
	fmt.Println("The client 4 has connected")
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
	
	// 4. Set the Content-Type header
	req.Header.Set("Content-Type", "application/json")
	
	// Execute the request
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