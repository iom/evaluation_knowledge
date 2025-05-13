# Evaluation Evidence Mapping for IOM

## Introduction

An **Evaluation Evidence Map** is a structured, visual tool that organizes what we know—and don’t know—about programs, policies, and interventions. Think of it as a research landscape that helps decision-makers quickly understand:

- Which interventions have been evaluated  
- Where they were implemented  
- What outcomes were observed  
- Where critical knowledge gaps remain  

This approach is especially valuable for evidence-informed project design, particularly when time or resources limit the ability to read through hundreds of individual evaluation reports (see [UNICEF Evidence Map example](https://evaluationreports.unicef.org/app/evaluation-evidence-gap-map.html)).

The goal of this exercise is to inform future evaluation design, guide strategic planning, and support the development of robust, evidence-based project proposals and strategic plan.

---

## Approach

### Defining the Focus

To ensure relevance, we begin by clarifying the scope and purpose of the mapping:

- **What types of interventions are we assessing?** (IOM programs, policies, and strategies, e.g., cash-based interventions, health services, community engagement)

- **Who are the target populations?**  (e.g., migrants, displaced persons, host communities)  

- **What outcomes matter most?**  (e.g., livelihood improvements, health outcomes, social integration - Aligned with the [IOM Strategic Results Framework](https://www.iom.int/iom-strategic-results-framework-srf))  

- **Who is the audience for this map?**   (e.g., policymakers, funders, researchers, program managers, donors)  

- **What are our key learning questions?**  (e.g., “What works best to maximize impact an effectivness?”)  

- **What level of evidence is required?**  (e.g., RCTs, quasi-experiments, observational studies)  

- **What variables are we tracking?** (e.g., intervention type, target group, outcomes, geography)

---

### Building the Knowledge Base

We have compiled a list of all publicly available [IOM Evaluation Reports](https://www.iom.int/evaluation-reports).

Each report will be analyzed to generate a structured **metadata record**, including:

- **What**: Title, summary, full-text link, evaluation type (formative, summative, impact), scope (strategy, policy, thematic, program, or project), and geographic coverage  
- **Who**: Conducting entity (IOM internal vs. external evaluators)  
- **How**: Methodology, study design, sample size, and data collection techniques  

These metadata and full-text documents will be convert the content of each report into a **embeddings vector database**, enabling fast, flexible, and AI-enhanced retrieval using advanced tools like [Hybrid Search](https://docs.lancedb.com/core/hybrid-search).

---

### Structured Information Extraction

We will create a set of plain-language questions, reflecting the entire IOM Results Framework.
Using AI tools, we will extract consistent and comparable data from each report:

- **Program details** (what was implemented)  
- **Context** (where and with whom)  
- **Design** (how it was studied)  
- **Findings** (what results were observed)  
- **Strength of evidence** (how reliable the findings are)

---

### Organize the Evidence

We will then run those questions through the vector database to generate answers based first on each evaluation.
This will allow to categorize the data within a structured framework:

1. **By intervention type** (e.g., skills training, psychosocial support)  
2. **By measured outcome** (e.g., employment, resilience, community cohesion)  
3. **By population** (e.g., migrants in transit, returnees, host communities)  
4. **By evidence quality** (e.g., robust vs. exploratory studies)

---

### Generate Actionable Insights with AI-Enabled Q&A

We will then re-run the same questions on the full corpus of Q&A previously generated, therefore accounting for all evidences that were gathered across all evaluations. 

This will allow us to quickly assess the evidence base and identify key insights, such as: _ "What types of interventions are most effective in improving livelihood outcomes for migrants in urban settings?"_

---

### Identify Patterns and Gaps

We’ll present the results across objectives, outcome, population and geography using intuitive, interactive visualizations:

- **Bubble maps** (bubble size = number of studies or sample size)  
- **Heatmaps** (showing concentration of evidence by topic or geography)  
- **Gap maps** (highlighting under-researched areas)

The evidence system will allow us to quickly highlight:

- ✅ Areas with strong, consistent evidence  
- ⚠️ Topics with mixed or conflicting findings  
- ❌ Critical gaps where no evidence exists  

*Example Insight:*  
> “Mentoring programs show consistent positive results for urban migrant youth, but there’s limited evidence for rural populations.”

---

## Deliverables

The final deliverables will include:

- A Q&A dataset that can be used:
  - As a reference for [content curation and evaluation](https://api.labelstud.io/tutorials/tutorials/evaluate-llm-responses)  
  - As a [knowledge base](https://docs.crewai.com/concepts/knowledge) for AI-enhanced project proposal systems  
  - As a training dataset for fine-tuning small language models via [Hugging Face](https://huggingface.co/)  

- A synthesis report identifying:
  - Research priorities  
  - High-risk areas for intervention  
  - Recommendations for future evaluations  

- A searchable online **visual evidence map** for ongoing use by IOM teams

