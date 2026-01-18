---
layout: default
permalink: /blog/
title: blog
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post">




  <ul class="post-list">

    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts %}
    {% else %}
      {% assign postlist = site.posts %}
    {% endif %}

    {% for post in postlist %}

    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <li>
      <div class="row">
        <!-- Image column -->
        <div class="col-sm-3">
          {% comment %} Extract image path from post content {% endcomment %}
          {% assign image_path = "" %}
          {% if post.content contains 'assets/img/9.jpg' %}
            {% assign image_path = 'assets/img/9.jpg' %}
          {% elsif post.content contains 'assets/img/gaussian-processes.png' %}
            {% assign image_path = 'assets/img/gaussian-processes.png' %}
          {% elsif post.content contains 'assets/img/robot-casino.jpg' %}
            {% assign image_path = 'assets/img/robot-casino.jpg' %}
          {% endif %}
          
          {% if image_path != "" %}
            <img class="card-img" src="{{ image_path | relative_url }}" style="object-fit: cover; height: 120px; width: 100%; border-radius: 0.25rem;" alt="{{ post.title }}">
          {% else %}
            <div style="height: 120px; width: 100%; background-color: #f8f9fa; border-radius: 0.25rem; display: flex; align-items: center; justify-content: center;">
              <span style="color: #6c757d;">No image</span>
            </div>
          {% endif %}
        </div>
        
        <!-- Content column -->
        <div class="col-sm-9">
          <h3>
            {% if post.redirect == blank %}
              <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {% elsif post.redirect contains '://' %}
              <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
              <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
              </svg>
            {% else %}
              <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
            {% endif %}
          </h3>
          <p>{{ post.description }}</p>
          <p class="post-meta">
            {{ read_time }} min read &nbsp; &middot; &nbsp;
            {{ post.date | date: '%B %d, %Y' }}
            {% if post.external_source %}
            &nbsp; &middot; &nbsp; {{ post.external_source }}
            {% endif %}
          </p>
        </div>
      </div>
    </li>

    {% endfor %}

  </ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>
