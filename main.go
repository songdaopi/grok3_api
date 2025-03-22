package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/andybalholm/brotli"
	"github.com/google/uuid"
)

// GrokClient defines a client for interacting with the Grok 3 Web API.
// It encapsulates the API endpoints, HTTP headers, and configuration flags.
type GrokClient struct {
	headers        map[string]string // HTTP headers for API requests
	isReasoning    bool              // Flag for using reasoning model
	enableSearch   bool              // Flag for searching in the Web
	uploadMessage  bool              // Flag for uploading the message as a file
	keepChat       bool              // Flag to preserve chat history
	ignoreThinking bool              // Flag to exclude thinking tokens in responses
}

// NewGrokClient creates a new instance of GrokClient with the provided cookies and configuration flags.
func NewGrokClient(cookie string, isReasoning bool, enableSearch bool, uploadMessage bool, keepChat bool, ignoreThinking bool) *GrokClient {
	return &GrokClient{
		headers: map[string]string{
			"accept":             "*/*",
			"accept-encoding":    "gzip, deflate, br, zstd",
			"accept-language":    "en-US;q=0.9,en;q=0.8",
			"content-type":       "application/json",
			"host":               "grok.com",
			"origin":             "https://grok.com",
			"dnt":                "1",
			"priority":           "u=1, i",
			"referer":            "https://grok.com/",
			"sec-ch-ua":          `"Not:A-Brand";v="24", "Chromium";v="134"`,
			"sec-ch-ua-mobile":   "?0",
			"sec-ch-ua-platform": `"Linux"`,
			"sec-fetch-dest":     "empty",
			"sec-fetch-mode":     "cors",
			"sec-fetch-site":     "same-origin",
			"user-agent":         "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
			"cookie":             cookie,
		},
		isReasoning:    isReasoning,
		enableSearch:   enableSearch,
		uploadMessage:  uploadMessage,
		keepChat:       keepChat,
		ignoreThinking: ignoreThinking,
	}
}

type ToolOverrides struct {
	ImageGen     bool `json:"imageGen"`
	TrendsSearch bool `json:"trendsSearch"`
	WebSearch    bool `json:"webSearch"`
	XMediaSearch bool `json:"xMediaSearch"`
	XPostAnalyze bool `json:"xPostAnalyze"`
	XSearch      bool `json:"xSearch"`
}

// preparePayload constructs the request payload for the Grok 3 Web API based on the given message and reasoning flag.
func (c *GrokClient) preparePayload(message string, fileId string) map[string]any {
	var toolOverrides any = ToolOverrides{}
	if c.enableSearch {
		toolOverrides = map[string]any{}
	}

	fileAttachments := []string{}
	if fileId != "" {
		fileAttachments = []string{fileId}
	}

	return map[string]any{
		"disableSearch":             false,
		"enableImageGeneration":     true,
		"enableImageStreaming":      true,
		"enableSideBySide":          true,
		"fileAttachments":           fileAttachments,
		"forceConcise":              false,
		"imageAttachments":          []string{},
		"imageGenerationCount":      2,
		"isPreset":                  false,
		"isReasoning":               c.isReasoning,
		"message":                   message,
		"modelName":                 "grok-3",
		"returnImageBytes":          false,
		"returnRawGrokInXaiRequest": false,
		"sendFinalMetadata":         true,
		"temporary":                 !c.keepChat,
		"toolOverrides":             toolOverrides,
		"webpageUrls":               []string{},
	}
}

// getModelName returns the appropriate model name based on the isReasoning flag.
func (c *GrokClient) getModelName() string {
	if c.isReasoning {
		return grok3ReasoningModelName
	} else {
		return grok3ModelName
	}
}

// RequestBody represents the structure of the JSON body expected in POST requests to the /v1/chat/completions endpoint,
// following the OpenAI API format. It includes fields for model selection, messages, streaming option, and additional specific options.
type RequestBody struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	Stream           bool   `json:"stream"`
	GrokCookies      any    `json:"grokCookies,omitempty"`   // A single cookie(string), or a list of cookie([]string)
	CookieIndex      uint   `json:"cookieIndex,omitempty"`   // Start from 1, 0 means auto-select cookies in turn
	EnableSearch     int    `json:"enableSearch,omitempty"`  // > 0 is true, == 0 is false
	UploadMessage    int    `json:"uploadMessage,omitempty"` // > 0 is true, == 0 is false
	TextBeforePrompt string `json:"textBeforePrompt,omitempty"`
	TextAfterPrompt  string `json:"textAfterPrompt,omitempty"`
	KeepChat         int    `json:"keepChat,omitempty"`       // > 0 is true, == 0 is false
	IgnoreThinking   int    `json:"ignoreThinking,omitempty"` // > 0 is true, == 0 is false
}

// ResponseToken represents a single token response from the Grok 3 Web API.
type ResponseToken struct {
	Result struct {
		Response struct {
			Token      string `json:"token"`
			IsThinking bool   `json:"isThinking"`
		} `json:"response"`
	} `json:"result"`
}

// ModelData represents model metadata for OpenAI-compatible response.
type ModelData struct {
	Id       string `json:"id"`
	Object   string `json:"object"`
	Owned_by string `json:"owned_by"`
}

// ModelList contains available models for OpenAI-compatible endpoint.
type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

// UploadFileRequest represents the request for uploading a file.
type UploadFileRequest struct {
	Content      string `json:"content"`
	FileMimeType string `json:"fileMimeType"`
	FileName     string `json:"fileName"`
}

// UploadFileResponse represents the response for uploading a file.
type UploadFileResponse struct {
	FileMetadataId string `json:"fileMetadataId"`
}

const (
	newChatUrl    = "https://grok.com/rest/app-chat/conversations/new" // Endpoint for creating new conversations
	uploadFileUrl = "https://grok.com/rest/app-chat/upload-file"       // Endpoint for uploading files

	grok3ModelName          = "grok-3"
	grok3ReasoningModelName = "grok-3-reasoning"

	completionsPath = "/v1/chat/completions"
	listModelsPath  = "/v1/models"

	messageCharsLimit = 50000

	defaultBeforePromptText    = "For the data below, entries with '[[system]]' are system information, entries with '[[assistant]]' are messages you have previously sent, entries with '[[user]]' are messages sent by the user. You need to respond to the user's last message accordingly based on the corresponding data."
	defaultUploadMessagePrompt = "Follow the instructions in the attached file to respond."
)

// Global configuration variables set.
var (
	apiToken         *string
	grokCookies      []string
	textBeforePrompt *string
	textAfterPrompt  *string
	keepChat         *bool
	ignoreThinking   *bool
	charsLimit       *uint
	httpProxy        *string
	httpClient       = &http.Client{Timeout: 30 * time.Minute}
	nextCookieIndex  = struct { // Thread-safe cookie rotation
		sync.Mutex
		index uint // Start from 0
	}{}
)

// decompressBody decompresses the response body.
func decompressBody(resp *http.Response) (io.ReadCloser, error) {
	switch resp.Header.Get("content-encoding") {
	case "br":
		return io.NopCloser(brotli.NewReader(resp.Body)), nil
	case "gzip":
		return gzip.NewReader(resp.Body)
	case "":
		return resp.Body, nil
	default:
		return nil, fmt.Errorf("unknown response encoding: %s", resp.Header.Get("content-encoding"))
	}

}

// doRequest sends the HTTP request.
func (c *GrokClient) doRequest(method string, url string, payload any) (*http.Response, error) {
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest(method, url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, err := decompressBody(resp)
		if err != nil {
			return nil, err
		}
		defer respBody.Close()
		body, err := io.ReadAll(respBody)
		if err != nil {
			return nil, fmt.Errorf("the Grok API error: %s", resp.Status)
		}
		return nil, fmt.Errorf("the Grok API error: %s, response body: %s", resp.Status, string(body)[:min(len(body), 128)])
	}

	return resp, nil
}

func (c *GrokClient) uploadMessageAsFile(message string) (*UploadFileResponse, error) {
	content := base64.StdEncoding.EncodeToString([]byte(message))
	payload := UploadFileRequest{
		Content:      content,
		FileMimeType: "text/plain",
		FileName:     uuid.New().String() + ".txt",
	}
	log.Println("Uploading the message as a file")
	resp, err := c.doRequest(http.MethodPost, uploadFileUrl, payload)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := decompressBody(resp)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()
	body, err := io.ReadAll(respBody)
	if err != nil {
		return nil, fmt.Errorf("uploading file error: %d %s", resp.StatusCode, resp.Status)
	}
	response := &UploadFileResponse{}
	err = json.Unmarshal(body, response)
	if err != nil {
		return nil, fmt.Errorf("parsing json error: %s", string(body))
	}
	if response.FileMetadataId == "" {
		return nil, fmt.Errorf("uploading file error: empty `FileMetadataId`")
	}

	return response, nil
}

// sendMessage sends a message to the Grok 3 Web API and returns the response body as an io.ReadCloser.
// If stream is true, it returns the streaming response; otherwise, it reads the entire response.
func (c *GrokClient) sendMessage(message string) (*http.Response, error) {
	fileId := ""
	if c.uploadMessage || (len(message) > int(*charsLimit) && utf8.RuneCountInString(message) > int(*charsLimit)) {
		uploadResp, err := c.uploadMessageAsFile(message)
		if err != nil {
			return nil, err
		}
		fileId = uploadResp.FileMetadataId
		message = defaultUploadMessagePrompt
	}

	payload := c.preparePayload(message, fileId)
	resp, err := c.doRequest(http.MethodPost, newChatUrl, payload)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

type OpenAIChatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIChatCompletionChunkChoice struct {
	Index        int                         `json:"index"`
	Delta        OpenAIChatCompletionMessage `json:"delta"`
	FinishReason string                      `json:"finish_reason"`
}

// OpenAIChatCompletionChunk represents the streaming response format for OpenAI.
type OpenAIChatCompletionChunk struct {
	ID      string                            `json:"id"`
	Object  string                            `json:"object"`
	Created int64                             `json:"created"`
	Model   string                            `json:"model"`
	Choices []OpenAIChatCompletionChunkChoice `json:"choices"`
}

type OpenAIChatCompletionChoice struct {
	Index        int                         `json:"index"`
	Message      OpenAIChatCompletionMessage `json:"message"`
	FinishReason string                      `json:"finish_reason"`
}

type OpenAIChatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIChatCompletion represents the non-streaming response format for OpenAI.
type OpenAIChatCompletion struct {
	ID      string                       `json:"id"`
	Object  string                       `json:"object"`
	Created int64                        `json:"created"`
	Model   string                       `json:"model"`
	Choices []OpenAIChatCompletionChoice `json:"choices"`
	Usage   OpenAIChatCompletionUsage    `json:"usage"`
}

// parseGrok3StreamingJson parses the streaming response from Grok 3.
func (c *GrokClient) parseGrok3StreamingJson(stream io.Reader, handler func(respToken string)) {
	// Read and process the streaming response from Grok 3
	isThinking := false
	decoder := json.NewDecoder(stream)
	for {
		var token ResponseToken
		err := decoder.Decode(&token)
		if err == io.EOF {
			break
		} else if err != nil {
			log.Printf("Parsing json error: %v", err)
			break
		}

		respToken := token.Result.Response.Token
		// Handle thinking tokens based on configuration
		if c.ignoreThinking && token.Result.Response.IsThinking {
			continue
		} else if token.Result.Response.IsThinking {
			if !isThinking {
				respToken = "<think>\n" + respToken
			}
			isThinking = true
		} else if isThinking {
			respToken = respToken + "\n</think>\n\n"
			isThinking = false
		}

		if respToken != "" {
			handler(respToken)
		}
	}
}

// createOpenAIStreamingResponse returns an HTTP handler that converts the Grok 3 streaming response to OpenAI's streaming format
// and writes it to the response writer.
func (c *GrokClient) createOpenAIStreamingResponse(grokStream io.Reader) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set up headers for streaming response
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, logPrintf("Streaming unsupported"), http.StatusInternalServerError)
			return
		}

		completionID := "chatcmpl-" + uuid.New().String()

		// Create and send the initial chunk
		startChunk := OpenAIChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   c.getModelName(),
			Choices: []OpenAIChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: OpenAIChatCompletionMessage{
						Role: "assistant",
					},
					FinishReason: "",
				},
			},
		}
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(startChunk))
		flusher.Flush()

		c.parseGrok3StreamingJson(grokStream, func(respToken string) {
			// Send token as a chunk in OpenAI format
			chunk := OpenAIChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   c.getModelName(),
				Choices: []OpenAIChatCompletionChunkChoice{
					{
						Index: 0,
						Delta: OpenAIChatCompletionMessage{
							Content: respToken,
						},
						FinishReason: "",
					},
				},
			}
			fmt.Fprintf(w, "data: %s\n\n", mustMarshal(chunk))
			flusher.Flush()
		})

		// Send the final chunk after streaming ends
		finalChunk := OpenAIChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   c.getModelName(),
			Choices: []OpenAIChatCompletionChunkChoice{
				{
					Index:        0,
					Delta:        OpenAIChatCompletionMessage{},
					FinishReason: "stop",
				},
			},
		}
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(finalChunk))
		flusher.Flush()

		// Finish the stream
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()

	}
}

// createOpenAIFullResponse returns an HTTP handler that reads the full Grok 3 response, converts it to OpenAI's format,
// and writes it to the response writer.
func (c *GrokClient) createOpenAIFullResponse(grokFull io.Reader) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var fullResponse strings.Builder
		c.parseGrok3StreamingJson(grokFull, func(respToken string) {
			fullResponse.WriteString(respToken)
		})

		openAIResponse := c.createOpenAIFullResponseBody(fullResponse.String())
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(openAIResponse); err != nil {
			http.Error(w, logPrintf("Encoding response error: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// createOpenAIFullResponseBody creates the OpenAI response body for non-streaming requests.
func (c *GrokClient) createOpenAIFullResponseBody(content string) OpenAIChatCompletion {
	return OpenAIChatCompletion{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   c.getModelName(),
		Choices: []OpenAIChatCompletionChoice{
			{
				Index: 0,
				Message: OpenAIChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: OpenAIChatCompletionUsage{
			PromptTokens:     -1,
			CompletionTokens: -1,
			TotalTokens:      -1,
		},
	}
}

// mustMarshal serializes the given value to a JSON string, panicking on error.
func mustMarshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// logPrintf prints the message to the standard logger and returns the string.
func logPrintf(format string, a ...any) string {
	log.Printf(format, a...)

	return fmt.Sprintf(format, a...)
}

// getCookieIndex selects the next cookie index in a round-robin fashion for load balancing or rotation.
// If cookieIndex is 0 or out of range, it uses the next index in sequence (len must be > 0).
func getCookieIndex(len int, cookieIndex uint) uint {
	if cookieIndex == 0 || cookieIndex > uint(len) {
		nextCookieIndex.Lock()
		defer nextCookieIndex.Unlock()
		index := nextCookieIndex.index
		nextCookieIndex.index = (nextCookieIndex.index + 1) % uint(len)

		return index % uint(len)
	} else {
		return cookieIndex - 1
	}
}

// handleChatCompletion handles incoming POST requests to /v1/chat/completions, authenticates the request,
// parses the request body, interacts with the Grok 3 API, and sends the response in OpenAI's format.
func handleChatCompletion(w http.ResponseWriter, r *http.Request) {
	// Log the request details
	log.Printf("Request from %s for %s", r.RemoteAddr, completionsPath)

	if r.URL.Path != completionsPath {
		http.Error(w, logPrintf("Requested Path %s Not Found", r.URL.Path), http.StatusNotFound)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, logPrintf("Method %s Not Allowed", r.Method), http.StatusMethodNotAllowed)
		return
	}

	// Authenticate the request
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
		http.Error(w, logPrintf("Unauthorized: Bearer token required"), http.StatusUnauthorized)
		return
	}

	token := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))
	if token != *apiToken {
		http.Error(w, logPrintf("Unauthorized: Invalid token"), http.StatusUnauthorized)
		return
	}

	// Parse the request body
	body := RequestBody{EnableSearch: -1, KeepChat: -1, IgnoreThinking: -1}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		http.Error(w, logPrintf("Bad Request: Invalid JSON"), http.StatusBadRequest)
		return
	}

	// Select the appropriate cookie
	var cookie string
	var cookieIndex uint
	if body.GrokCookies != nil {
		if ck, ok := body.GrokCookies.(string); ok {
			cookie = ck
		} else if list, ok := body.GrokCookies.([]any); ok {
			if len(list) > 0 {
				cookieIndex = getCookieIndex(len(list), body.CookieIndex)
				if ck, ok := list[cookieIndex].(string); ok {
					cookie = ck
				} else {
					http.Error(w, logPrintf("Error: Invalid Grok 3 cookie"), http.StatusBadRequest)
					return
				}
			}
		}
	}
	cookie = strings.TrimSpace(cookie)
	if cookie == "" && len(grokCookies) > 0 {
		cookieIndex = getCookieIndex(len(grokCookies), body.CookieIndex)
		cookie = grokCookies[cookieIndex]
	}
	cookie = strings.TrimSpace(cookie)
	if cookie == "" {
		http.Error(w, logPrintf("Error: No Grok 3 cookie"), http.StatusBadRequest)
		return
	}

	messages := body.Messages
	if len(messages) == 0 {
		http.Error(w, logPrintf("Bad Request: No messages provided"), http.StatusBadRequest)
		return
	}

	var beforePromptText string
	var afterPromptText string
	if body.TextBeforePrompt != "" {
		beforePromptText = body.TextBeforePrompt
	} else {
		beforePromptText = *textBeforePrompt
	}
	if body.TextAfterPrompt != "" {
		afterPromptText = body.TextAfterPrompt
	} else {
		afterPromptText = *textAfterPrompt
	}

	// Construct the message to send to Grok 3
	var messageBuilder strings.Builder
	fmt.Fprintln(&messageBuilder, beforePromptText)
	for _, msg := range messages {
		fmt.Fprintf(&messageBuilder, "\n[[%s]]\n", msg.Role)
		if content, ok := msg.Content.(string); ok {
			messageBuilder.WriteString(content)
		} else if messages, ok := msg.Content.([]any); ok && msg.Role == "user" {
			for _, message := range messages {
				if text, ok := message.(map[string]any); ok {
					messageType := text["type"]
					if ty, ok := messageType.(string); ok && ty == "text" {
						if t, ok := text["text"].(string); ok {
							fmt.Fprintln(&messageBuilder, t)
						} else {
							http.Error(w, logPrintf("Bad Request: Unsupported message type"), http.StatusBadRequest)
							return
						}
					} else {
						http.Error(w, logPrintf("Bad Request: Unsupported message type"), http.StatusBadRequest)
						return
					}
				} else {
					http.Error(w, logPrintf("Bad Request: Unsupported message type"), http.StatusBadRequest)
					return
				}
			}
		} else {
			http.Error(w, logPrintf("Bad Request: Unsupported message type"), http.StatusBadRequest)
			return
		}
	}
	fmt.Fprintf(&messageBuilder, "\n%s", afterPromptText)

	// Determine configuration flags
	isReasoning := false
	if strings.TrimSpace(body.Model) == grok3ReasoningModelName {
		isReasoning = true
	}

	enableSearch := false
	if body.EnableSearch > 0 {
		enableSearch = true
	}

	uploadMessage := false
	if body.UploadMessage > 0 {
		uploadMessage = true
	}

	keepConversation := false
	if body.KeepChat > 0 {
		keepConversation = true
	} else if body.KeepChat < 0 {
		keepConversation = *keepChat
	}

	ignoreThink := false
	if body.IgnoreThinking > 0 {
		ignoreThink = true
	} else if body.IgnoreThinking < 0 {
		ignoreThink = *ignoreThinking
	}

	// Initialize GrokClient with selected options
	grokClient := NewGrokClient(cookie, isReasoning, enableSearch, uploadMessage, keepConversation, ignoreThink)
	log.Printf("Use the cookie with index %d to request Grok 3 Web API", cookieIndex+1)
	// Send the message to Grok 3 Web API
	resp, err := grokClient.sendMessage(messageBuilder.String())
	if err != nil {
		http.Error(w, logPrintf("Error: %v", err), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	respBody, err := decompressBody(resp)
	if err != nil {
		http.Error(w, logPrintf("Error: %v", err), http.StatusInternalServerError)
		return
	}
	defer respBody.Close()

	// Handle the response based on streaming option
	if body.Stream {
		grokClient.createOpenAIStreamingResponse(respBody)(w, r)
	} else {
		grokClient.createOpenAIFullResponse(respBody)(w, r)
	}
	_, _ = io.ReadAll(respBody)
}

// listModels handles GET requests to /v1/models, returning a list of available models in OpenAI's format.
func listModels(w http.ResponseWriter, r *http.Request) {
	// Log the request details
	log.Printf("Request from %s for %s", r.RemoteAddr, listModelsPath)

	if r.URL.Path != listModelsPath {
		http.Error(w, logPrintf("Requested Path %s Not Found", r.URL.Path), http.StatusNotFound)
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, logPrintf("Method %s Not Allowed", r.Method), http.StatusMethodNotAllowed)
		return
	}

	list := ModelList{
		Object: "list",
		Data: []ModelData{
			{
				Id:       grok3ModelName,
				Object:   "model",
				Owned_by: "xAI",
			},
			{
				Id:       grok3ReasoningModelName,
				Object:   "model",
				Owned_by: "xAI",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(list); err != nil {
		http.Error(w, logPrintf("Encoding response error: %v", err), http.StatusInternalServerError)
		return
	}
}

// main parses command-line flags, sets up the HTTP server, and starts listening for requests.
func main() {
	// Define command-line flags
	apiToken = flag.String("token", "", "Authentication token (GROK3_AUTH_TOKEN)")
	cookie := flag.String("cookie", "", "Grok cookie(s) (GROK3_COOKIE)")
	cookieFile := flag.String("cookieFile", "", "A text file which contains Grok cookies line by line")
	textBeforePrompt = flag.String("textBeforePrompt", defaultBeforePromptText, "Text before the prompt")
	textAfterPrompt = flag.String("textAfterPrompt", "", "Text after the prompt")
	keepChat = flag.Bool("keepChat", false, "Retain the chat conversation")
	ignoreThinking = flag.Bool("ignoreThinking", false, "Ignore the thinking content while using the reasoning model")
	charsLimit = flag.Uint("charsLimit", messageCharsLimit, "Upload the message as a file if the count of its characters is greater than this limit")
	httpProxy = flag.String("httpProxy", "", "HTTP/SOCKS5 proxy")
	port := flag.Uint("port", 8180, "Server port")
	flag.Parse()

	// Validate port number
	if *port > 65535 {
		log.Fatalf("Server port %d is greater than 65535", *port)
	}

	// Set authentication token from flag or environment variable
	*apiToken = strings.TrimSpace(*apiToken)
	if *apiToken == "" {
		*apiToken = strings.TrimSpace(os.Getenv("GROK3_AUTH_TOKEN"))
		if *apiToken == "" {
			log.Fatal("Authentication token (GROK3_AUTH_TOKEN) is unset")
		}
	}

	// Set cookies from flag or environment variable
	*cookie = strings.TrimSpace(*cookie)
	if *cookie == "" {
		*cookie = strings.TrimSpace(os.Getenv("GROK3_COOKIE"))
	}
	if *cookie != "" {
		err := json.Unmarshal([]byte(*cookie), &grokCookies)
		if err != nil {
			grokCookies = []string{*cookie}
		}
	}
	// Get cookies from `cookieFile`
	if *cookieFile != "" {
		file, err := os.Open(*cookieFile)
		if err != nil {
			log.Fatalf("Open file %s error: %v", *cookieFile, err)
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			c := strings.TrimSpace(scanner.Text())
			if c != "" {
				grokCookies = append(grokCookies, c)
			}
		}
		if err = scanner.Err(); err != nil {
			log.Fatalf("Reading file %s error: %v", *cookieFile, err)
		}
	}

	// Configure HTTP client with proxy if provided
	*httpProxy = strings.TrimSpace(*httpProxy)
	if *httpProxy != "" {
		proxyURL, err := url.Parse(*httpProxy)
		if err == nil {
			httpClient.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyURL),
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				ForceAttemptHTTP2:   true,
				MaxIdleConns:        10,
				IdleConnTimeout:     600 * time.Second,
				TLSHandshakeTimeout: 20 * time.Second,
			}
		} else {
			log.Fatalf("Parsing HTTP/SOCKS5 proxy error：%v", err)
		}
	}

	// Register HTTP handlers
	http.HandleFunc(completionsPath, handleChatCompletion)
	http.HandleFunc(listModelsPath, listModels)
	log.Printf("Server starting on :%d", *port)

	// Start the HTTP server
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}
