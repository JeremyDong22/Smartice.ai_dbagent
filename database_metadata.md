# Database Metadata for Gemini 2.0 Flash Lite

**Database Type:** PostgreSQL (running on Supabase, supports pgvector extension)

**Core Business Scenario:** Storing and analyzing restaurant brand information from Dianping (including rankings) and related high-engagement (>500 likes) posts from Xiaohongshu, with support for semantic search based on post content.

---

## Table Structures and Descriptions

### Table: `brand`

* **Description:** Stores basic information about restaurant brands obtained from Dianping. Contains distinct brand names found in the `dzdpdata` table.
* **Primary Key:** `brand_id`
* **Columns:**
    * `brand_id` (int8, **Primary Key**): Unique identifier for the brand.
    * `name` (text): The name of the brand. This name can be used to query ranking information for this brand in the `dzdpdata` table by filtering the `dzdpdata.品牌` column.

### Table: `dzdpdata` (Dianping Data)

* **Description:** Stores brand ranking information obtained from Dianping. Data is organized by city, list type, and date. **When querying rankings, ALWAYS filter for the most recent `create_date`.**
* **Primary Key:** Composite (`榜单`, `品牌`, `create_date`)
* **Columns:**
    * `榜单` (text): The name of the ranking list, e.g., "XX市热门榜" (main list) or specific category lists like "创意菜", "咖啡", "烧烤". Part of the composite primary key.
    * `排名` (int4): The brand's rank within this list (`榜单`) on this date (`create_date`).
    * `店铺名称` (text): The full name of the specific shop as displayed on Dianping (may include address details, parentheses, etc.).
    * `品牌` (text): The name of the brand this shop belongs to (cleaned/derived from `店铺名称`). Part of the composite primary key. Use values from `brand.name` to filter this column when searching for a specific brand's ranking.
    * `评分` (numeric): The shop's rating on Dianping.
    * `位置` (text): The shop's address or location description.
    * `细分榜单` (text): **Ignore this column; it holds little value.** The relevant list name is already in the `榜单` column.
    * `价格` (int4): **Average price per person (in CNY) for dining at this shop.**
    * `城市` (text): The city to which this ranking belongs.
    * `create_date` (date): The date when this ranking data was obtained. Part of the composite primary key. **Crucial for finding the latest rankings.**

### Table: `posts`

* **Description:** Stores information about posts from Xiaohongshu that are related to brands in the `brand` table and have more than 500 likes.
* **Primary Key:** `post_id`
* **Columns:**
    * `post_id` (text, **Primary Key**): Unique identifier for the Xiaohongshu post.
    * `likes` (int4): Number of likes for the post (all entries in this table have > 500 likes).
    * `title` (text): The title of the post.
    * `author` (text): The author's nickname or ID.
    * `publish_date` (date): The date the post was published.
    * `content` (text): The original text content of the post.
    * `images` (text): **A comma-separated (`,`) list of public URLs for the post's images, stored within a single text field.** (Not a `text[]` array).
    * `collections` (int4): Number of times the post was collected/saved.
    * `comments` (int4): Number of comments on the post.
    * `vec_content` (vector[1536]): **A 1536-dimensional vector embedding of the post's text content (`content`), generated using OpenAI's `text-embedding-3-small` model.** Used for semantic similarity searches. Use the `<=>` operator for cosine distance similarity search (smaller distance means more similar).
    * `is_related` (bool): **Ignore this column.**

### Table: `post_brand` (Junction Table)

* **Description:** Connects the `posts` and `brand` tables, representing their many-to-many relationship. A post can mention multiple brands, and a brand can appear in multiple posts.
* **Primary Key:** Composite (`post_id`, `brand_id`) [Assumed standard practice]
* **Foreign Keys:**
    * `post_id` (text) references `posts.post_id`
    * `brand_id` (int8) references `brand.brand_id`
* **Columns:**
    * `post_id` (text): The ID of the related post.
    * `brand_id` (int8): The ID of the related brand.

---

## Relationships Summary

* **`posts` <-> `brand`:** Many-to-Many relationship via `post_brand`.
* **`brand.name` -> `dzdpdata.品牌`:** Logical relationship. Use `brand.name` to filter `dzdpdata` on the `品牌` column to find rankings for a specific brand.

---

## Important Notes for LLM (Gemini 2.0 Flash Lite)

1.  **Latest Rankings:** When asked for any ranking information from `dzdpdata`, **always** retrieve the records with the most recent `create_date`. This typically requires ordering by `create_date` descending and taking the top result(s) per group, or using window functions.
2.  **Brand Querying:** When a user mentions a brand name (found in `brand.name`), use that name to filter the `品牌` column in the `dzdpdata` table to get rankings.
3.  **Vector Search:** For queries about post content similarity (e.g., "find posts similar to XXX"):
    * Obtain the 1536-dimensional vector embedding for the user's query text ("XXX") using the `text-embedding-3-small` model.
    * Use `posts.vec_content <=> '{query_vector}'` in the `ORDER BY` clause of your SQL query, limiting the results (`LIMIT`). `<=>` is the cosine distance operator from pgvector.
4.  **Chinese Column Names:** The `dzdpdata` table uses Chinese column names (`榜单`, `排名`, `店铺名称`, `品牌`, `评分`, `位置`, `价格`, `城市`). Ensure these are correctly quoted (usually with double quotes `""`) in the generated SQL query. Example: `SELECT "排名" FROM dzdpdata WHERE "品牌" = 'SomeBrand';`
5.  **Ignored Columns:** Do not use or refer to the `dzdpdata.细分榜单` or `posts.is_related` columns.
6.  **Image Handling:** The `posts.images` column is a single text field containing comma-separated URLs. If individual URLs are needed, string manipulation (like `string_to_array` in SQL or application-level processing) might be required after retrieving the data.