bright_query_gen = """You are tasked with generating reasoning-intensive queries for retrieval tasks. These queries should require intensive reasoning to identify that the provided document content is relevant—simple keyword matching or semantic similarity should NOT be sufficient.

## What Makes a Query "Reasoning-Intensive"?

A reasoning-intensive query requires one or more of these reasoning types to connect it to the document:

1. **Deductive Reasoning**: The document describes a general principle, theorem, or mechanism that can be applied to solve a specific problem or explain a specific scenario in the query.

2. **Analogical Reasoning**: The document presents a parallel situation that uses similar underlying logic, algorithms, or approaches, even if the surface-level context appears different.

3. **Causal Reasoning**: The document provides the cause or explanation for a problem/phenomenon described in the query, or vice versa.

4. **Analytical Reasoning**: The document provides critical concepts, components, or knowledge that form essential pieces of a reasoning chain needed to solve the query problem.

## Requirements for Your Generated Query

Given the document content, generate a query that:

**MUST have LOW lexical overlap** - Avoid copying phrases or technical terms directly from the document  
**MUST have LOW semantic similarity** - Don't just paraphrase the document; create a genuinely different scenario  
**MUST require reasoning** - The connection should require understanding underlying principles, not surface matching  
**Should be realistic** - Frame it as a natural question someone would actually ask (troubleshooting, problem-solving, learning)  
**Should be specific and detailed** - Include context, constraints, or details that make it concrete  
**Should be standalone** - The query should be understandable without seeing the document  

## Techniques to Ensure Reasoning-Intensive Connection

- **Ground in different domains**: If the document is about plant biology, ask about tree maintenance
- **Use different terminology**: If the document mentions "soluble salts," the query could discuss "dissolved minerals"
- **Present specific scenarios**: Instead of asking "what is X?", describe a situation where X is the underlying cause
- **Focus on application**: Ask how to solve a problem where the document's principle applies
- **Mask the connection**: Make it non-obvious that the document's concept is what's needed

Based on the provided document content, generate a single reasoning-intensive query that:
1. Would realistically require this document to answer
2. Has minimal keyword/semantic overlap with the document  
3. Requires genuine reasoning to identify the document as relevant
4. Is concrete, specific, and well-contextualized

Output only the query text, nothing else."""