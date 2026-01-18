---
layout: default
permalink: /contact/
title: contact
nav: true
nav_order: 7
---

<div class="post">
  <header class="post-header">
    <h1 class="post-title">{{ page.title | capitalize }}</h1>
  </header>

  <article class="post-content">
    <p>Get in touch with me using the form below:</p>
    
    <!-- modify this form HTML and place wherever you want your form -->
    <form
      action="https://formspree.io/f/xgooaelo"
      method="POST"
    >
      <label>
        Your email:
        <input type="email" name="email">
      </label>
      <label>
        Your message:
        <textarea name="message"></textarea>
      </label>
      <!-- your other form fields go here -->
      <button type="submit">Send</button>
    </form>
  </article>
</div>