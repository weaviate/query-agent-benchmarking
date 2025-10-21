simple_query_gen_prompt = """You are tasked with generating simple queries that this document contains the answer to.

Output only the query text, nothing else."""

bright_query_gen = """You are tasked with generating reasoning-intensive queries for retrieval tasks. These queries should require intensive reasoning to identify that the provided document content is relevant—simple keyword matching or semantic similarity should NOT be sufficient.

## What Makes a Query "Reasoning-Intensive"?

Here are some examples to guide your understanding:

Example 1:

If the brain has no pain receptors, how come you can get a headache? I've read many years ago in books, that the brain has no nerves on it, and if someone was touching your brain, you couldn't feel a thing. Just two days before now, I had a very bad migraine, due to a cold. It's become better now, but when I had it I felt my head was going to literally split in half, as the pain was literally coming from my brain. So it lead me to the question: How come people can get headaches if the brain has no nerves?

Example 2:

Why are fearful stimuli more powerful at night? For example, horror movies appear to be scarier when viewed at night than during broad day light. Does light have any role in this phenomenon? Are there changes in hormones at night versus during the day that makes fear stronger?

Example 3:

Is monogamy a human innate behaviour? As the question states, got curious and I was wondering if monogamy is an innate human behaviour or is it because of how we built society (religion, traditions, etc.)? Let's say we go back in time, would we see humans settling down with a single partner at a time and caring for their children as a couple for life or would they reproduce with several leaving the mothers with their children? Thanks!

Example 4:

Do animals exhibit handedness (paw-ness?) preference? I have been observing my cat and found that when confronted with an unknown item, she will always use her front left paw to touch it. This has me wondering if animals exhibit handedness like humans do? (and do I have a left handed cat?) One note of importance is that with an unknown item, her approach is always identical, so possibly using the left paw means allowing a fast possible exit based on how she positions her body. This question is related to Are there dextral/sinistral higher animals?. However, I question the "paw-ness" as a consequence of how the cat is approaching new items (to be ready to flee), whereas the other question remarks about the high number of "right-pawed" dogs and questions the influence of people for this preference.

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