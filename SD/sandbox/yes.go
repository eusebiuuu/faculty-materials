package sandbox

func Map(in chan []string) chan Pair {
	out := make(chan Pair)

	go func() {
		defer close(out)
		for wordsList := range in {
			var count = 0
			for _, word := range wordsList {
				if validWord(word) {
					count++
				}
			}
			out <- Pair{count, len(wordsList)}
		}
	}()
	return out
}