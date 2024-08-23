package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/exp/rand"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func readFile(fileName string) (string, error) {
	content, err := ioutil.ReadFile(fileName)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func mapCharToIdx(vocab string) map[string]int {
	charToIdx := map[string]int{}
	for i, char := range vocab {
		charToIdx[string(char)] = i
	}
	return charToIdx
}

func encodeString(text string, charToIdx map[string]int) []int {
	encodedText := []int{}
	for _, char := range text {
		encodedText = append(encodedText, charToIdx[string(char)])
	}
	return encodedText
}

func get_batch(split string, train_data, val_data []int, batch_size, block_size int) (*tensor.Dense, *tensor.Dense) {
	var data []int
	if split == "train" {
		data = train_data
	} else {
		data = val_data
	}

	rand.Seed(uint64(time.Now().UnixNano()))

	ix := make([]int, batch_size)
	for i := range ix {
		ix[i] = rand.Intn(len(data) - block_size)
	}

	xData := make([]float32, batch_size*block_size)
	yData := make([]float32, batch_size*block_size)

	for i := 0; i < batch_size; i++ {
		for j := 0; j < block_size; j++ {
			xData[i*block_size+j] = float32(data[ix[i]+j])
			yData[i*block_size+j] = float32(data[ix[i]+j+1])
		}
	}

	x := tensor.New(tensor.WithShape(batch_size, block_size), tensor.Of(tensor.Float32), tensor.WithBacking(xData))
	y := tensor.New(tensor.WithShape(batch_size, block_size), tensor.Of(tensor.Float32), tensor.WithBacking(yData))

	return x, y
}

// Define the BigramLanguageModel structure
type BigramLanguageModel struct {
	vocabSize           int
	tokenEmbeddingTable *gorgonia.Node
	g                   *gorgonia.ExprGraph
}

// Constructor for the BigramLanguageModel
func NewBigramLanguageModel(vocabSize int, g *gorgonia.ExprGraph) *BigramLanguageModel {
	// Initialize the embedding matrix with random values
	tokenEmbeddingTable := gorgonia.NewMatrix(
		g,
		tensor.Float32,
		gorgonia.WithShape(vocabSize, vocabSize),
		gorgonia.WithName("tokenEmbeddingTable"),
		gorgonia.WithInit(gorgonia.GlorotU(1)),
	)

	return &BigramLanguageModel{
		vocabSize:           vocabSize,
		tokenEmbeddingTable: tokenEmbeddingTable,
		g:                   g,
	}
}

// Forward pass for the model
func (m *BigramLanguageModel) Forward(idx, targets *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	// Embedding lookup
	logits, err := gorgonia.Mul(m.tokenEmbeddingTable, idx)
	if err != nil {
		return nil, nil, err
	}

	var loss *gorgonia.Node
	if targets != nil {
		// Reshape logits and targets for cross-entropy loss
		B, T := idx.Shape()[0], idx.Shape()[1]
		C := m.vocabSize
		logitsReshaped := gorgonia.Must(gorgonia.Reshape(logits, tensor.Shape{B * T, C}))
		targetsReshaped := gorgonia.Must(gorgonia.Reshape(targets, tensor.Shape{B * T}))

		// Compute the cross-entropy loss (assuming you implement this)
		loss, err = CrossEntropy(logitsReshaped, targetsReshaped)
		if err != nil {
			return nil, nil, err
		}
	}

	return logits, loss, nil
}

// CrossEntropy calculates the cross-entropy loss
func CrossEntropy(logits, targets *gorgonia.Node) (*gorgonia.Node, error) {
	// Implementation of cross-entropy, for example using softmax and negative log-likelihood
	logProbs, err := gorgonia.LogSoftMax(logits)
	if err != nil {
		return nil, err
	}
	nll, err := gorgonia.Neg(logProbs)
	if err != nil {
		return nil, err
	}
	loss := gorgonia.Must(gorgonia.Mean(nll))
	return loss, nil
}

// Generate new tokens based on the current context
func (m *BigramLanguageModel) Generate(idx *gorgonia.Node, maxNewTokens int) (*gorgonia.Node, error) {
	for i := 0; i < maxNewTokens; i++ {
		// Forward pass to get logits
		logits, _, err := m.Forward(idx, nil)
		if err != nil {
			return nil, err
		}

		// Focus on the last time step
		lastLogits, err := gorgonia.Slice(logits, gorgonia.S(0, -1, m.vocabSize))
		if err != nil {
			return nil, err
		}

		// Apply softmax to get probabilities
		_ = gorgonia.Must(gorgonia.SoftMax(lastLogits)) // Currently not using probs

		// Sampling from the distribution
		// You will need to implement this or find a library function

		// Append sampled index to the running sequence (Implementation needed)
		// idx = gorgonia.Must(gorgonia.Concat(1, idx, nextIdx))
	}

	return idx, nil
}

// Main function to train and test the model
func main() {
	text, err := readFile("data.txt")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	vocabSet := map[string]struct{}{}
	for _, char := range text {
		vocabSet[string(char)] = struct{}{}
	}

	var vocabSlice []string
	for char := range vocabSet {
		vocabSlice = append(vocabSlice, char)
	}

	sort.Strings(vocabSlice)

	sortedVocabStr := strings.Join(vocabSlice, "")
	fmt.Println("Sorted Vocab: ", sortedVocabStr)

	stoi := mapCharToIdx(sortedVocabStr)
	encodedText := encodeString(text, stoi)

	trainSize := int(0.9 * float64(len(encodedText)))
	train_data := encodedText[:trainSize]
	val_data := encodedText[trainSize:]

	block_size := 8
	batch_size := 4

	x, y := get_batch("train", train_data, val_data, batch_size, block_size)

	// Initialize the Gorgonia graph
	g := gorgonia.NewGraph()

	// Convert tensors to Gorgonia nodes
	xNode := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(batch_size, block_size), gorgonia.WithValue(x))
	yNode := gorgonia.NewTensor(g, tensor.Float32, 2, gorgonia.WithShape(batch_size, block_size), gorgonia.WithValue(y))

	// Create the BigramLanguageModel
	vocabSize := len(vocabSlice)
	model := NewBigramLanguageModel(vocabSize, g)

	// Forward pass through the model
	_, loss, err := model.Forward(xNode, yNode)
	if err != nil {
		fmt.Println("Error in forward pass:", err)
		return
	}

	// Run the computation graph
	machine := gorgonia.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		fmt.Println("Error running the computation graph:", err)
		return
	}

	// Print the loss
	if loss != nil {
		fmt.Println("Loss:", loss.Value())
	}

	// Generate new text
	startIdx := gorgonia.NewMatrix(g, tensor.Int, gorgonia.WithShape(1, 1), gorgonia.WithInit(gorgonia.Zeroes()))
	generated, err := model.Generate(startIdx, 100)
	if err != nil {
		fmt.Println("Error generating text:", err)
		return
	}

	// Print the generated sequence
	fmt.Println("Generated sequence:", generated.Value())
}
