// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#include "Ravl/XMPP/LMConnection.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/CallMethodPtrs.hh"
#include "Ravl/XMLFactoryRegister.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  namespace XMPPN {

#if 0

    static void
    connection_auth_cb(LmConnection *connection,
                       gboolean success,
                       gpointer user_data)
    {
      if(success) {
        GError *error = NULL;
        LmMessage *m;

        g_print("LmSendAsync: Authenticated successfully\n");

        m = lm_message_new(recipient, LM_MESSAGE_TYPE_MESSAGE);
        lm_message_node_add_child(m->node, "body", message);

        if(!lm_connection_send(connection, m, &error)) {
          g_printerr("LmSendAsync: Failed to send message:'%s'\n",
                     lm_message_node_to_string(m->node));
        } else {
          g_print("LmSendAsync: Sent message:'%s'\n",
                  lm_message_node_to_string(m->node));
          test_success = TRUE;
        }

        lm_message_unref(m);
      } else {
        g_printerr("LmSendAsync: Failed to authenticate\n");
      }

      lm_connection_close(connection, NULL);
      g_main_loop_quit(main_loop);
    }
#endif

    static void InternalHandlerConnectionAuth(LmConnection *connection,
                                              gboolean success,
                                              gpointer user_data)
    {
      RavlAssert(user_data != 0);
      return reinterpret_cast<LMConnectionC *> (user_data)->CBConnectionAuth(connection, success);
    }

    static void InternalHandlerConnectionOpen(LmConnection *connection,
                                              gboolean success,
                                              gpointer user_data)
    {
      RavlAssert(user_data != 0);
      return reinterpret_cast<LMConnectionC *> (user_data)->CBConnectionOpen(connection, success);
    }

    static LmHandlerResult InternalMessageHandler(LmMessageHandler *handler,
                                                  LmConnection *connection,
                                                  LmMessage *message,
                                                  gpointer user_data)
    {
      RavlAssert(user_data != 0);
      return reinterpret_cast<LMConnectionC *> (user_data)->MessageHandler(handler, connection, message);
    }

    //! Constructor

    LMConnectionC::LMConnectionC(const char *server, const char *user, const char *password, const char *resource)
    : m_server(server),
    m_user(user),
    m_password(password),
    m_resource(resource),
    m_dumpRaw(true),
    m_sigTextMessage("", ""),
    m_context(0),
    m_mainLoop(0),
    m_conn(0),
    m_defaultHandler(0),
    m_isReady(false)
    {
      Init(true);
    }

    //! Factory constructor

    LMConnectionC::LMConnectionC(const XMLFactoryContextC &factory)
    : m_server(factory.AttributeString("server", "jabber.org").data()),
    m_user(factory.AttributeString("user", "auser").data()),
    m_password(factory.AttributeString("password", "apassword").data()),
    m_resource(factory.AttributeString("resource", "default").data()),
    m_dumpRaw(factory.AttributeBool("dumpRaw", false)),
    m_sigTextMessage("", ""),
    m_context(0),
    m_mainLoop(0),
    m_conn(0),
    m_defaultHandler(0),
    m_isReady(false)

    {
      Init(factory.AttributeBool("useOwnThread", true));
    }


    //! Destructor

    LMConnectionC::~LMConnectionC()
    {
      lm_connection_unregister_message_handler(m_conn, m_defaultHandler, LM_MESSAGE_TYPE_MESSAGE);
      lm_message_handler_unref(m_defaultHandler);
      lm_connection_close(m_conn, NULL);
      lm_connection_unref(m_conn);

      if(m_mainLoop != 0) {
        g_main_loop_unref(m_mainLoop);
      }
      g_main_context_unref(m_context);
    }


    //: Called when owner handles drop to zero.

    void LMConnectionC::ZeroOwners()
    {
      if(m_mainLoop != 0) {
        g_main_loop_quit(m_mainLoop);
      }
    }

    // GLib main event loop if needed.

    int LMConnectionC::MainLoop()
    {
      ONDEBUG(std::cerr << "Starting main loop \n");
      g_main_context_push_thread_default(m_context);

      g_main_loop_run(m_mainLoop);
      ONDEBUG(std::cerr << "Done main loop. \n");
      return 0;
    }

    //! Start the server

    int LMConnectionC::Init(bool inOwnThread)
    {
      // Make sure threads are enabled.
      g_thread_init(0);

      if(inOwnThread) {
        m_context = g_main_context_new();
        m_mainLoop = g_main_loop_new(m_context, false);
        m_conn = lm_connection_new_with_context(m_server.data(), m_context);
        LaunchThread(TriggerPtr(CBRefT(this), &LMConnectionC::MainLoop));
      } else {
        m_context = g_main_context_default();
        g_main_context_ref(m_context);
        m_conn = lm_connection_new_with_context(m_server.data(), m_context);
      }
      return true;
    }

    //! Start opening

    bool LMConnectionC::Open()
    {
      GError *error = NULL;
      std::string jid = m_user + "@" + m_server;
      lm_connection_set_jid(m_conn, jid.data());

#if 0
      if(!lm_connection_open_and_block(m_conn, &error)) {
        g_error("Connection failed to open: %s   Account:'%s' ", error->message, jid.data());
        return false;
      }

      if(!lm_connection_authenticate_and_block(m_conn, m_user.data(), m_password.data(), m_resource.data(), &error)) {
        g_error("Connection failed to authenticate: %s", error->message);
        return false;
      }
#endif
      lm_connection_set_keep_alive_rate(m_conn, 20);

      m_defaultHandler = lm_message_handler_new(&InternalMessageHandler, (gpointer)this, NULL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_MESSAGE,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_PRESENCE,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_IQ,
                                             LM_HANDLER_PRIORITY_NORMAL);

#if 1
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_STREAM,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_AUTH,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_CHALLENGE,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_RESPONSE,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_FAILURE,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_PROCEED,
                                             LM_HANDLER_PRIORITY_NORMAL);
      lm_connection_register_message_handler(m_conn, m_defaultHandler,
                                             LM_MESSAGE_TYPE_STARTTLS,
                                             LM_HANDLER_PRIORITY_NORMAL);
#endif
#if 0
      // Let server know we want messages.
      LmMessage *m = lm_message_new(NULL, LM_MESSAGE_TYPE_PRESENCE);
      if(!lm_connection_send(m_conn, m, &error)) {
        g_error("Connection failed to send presence message: %s",
                error->message);
      }
      lm_message_unref(m);
#else

#if 0
      if(1) {
        LmSSL *ssl;
        char *p;
        int i;

        lm_connection_set_port(connection,LM_CONNECTION_DEFAULT_PORT_SSL);

        for(i = 0, p = fingerprint; *p && *(p + 1); i++, p += 3) {
          expected_fingerprint[i] = (unsigned char) g_ascii_strtoull(p, NULL, 16);
        }

        ssl = lm_ssl_new(expected_fingerprint,
                         (LmSSLFunction) ssl_cb,
                         NULL, NULL);

        lm_ssl_use_starttls(ssl, TRUE, FALSE);

        lm_connection_set_ssl(connection, ssl);
        lm_ssl_unref(ssl);
      }
#endif


      if(!lm_connection_open(m_conn,
                             (LmResultFunction) & InternalHandlerConnectionOpen,
                             this, NULL, &error)) {
        g_printerr("LmSendAsync: Could not open a connection %s \n", error->message);
        return EXIT_FAILURE;
      }


#endif
      return true;
    }

    //! Test if we have a connection.

    bool LMConnectionC::IsConnected() const
    {
      if(m_conn == 0)
        return false;
      return lm_connection_is_open(m_conn);
    }

    //! Send a text message to someone.

    bool LMConnectionC::SendText(const char *to, const char *message)
    {
      GError *error = NULL;
      LmMessage *m = lm_message_new(to, LM_MESSAGE_TYPE_MESSAGE);
      lm_message_node_add_child(m->node, "body", message);
      bool ret = true;
      if(!lm_connection_send(m_conn, m, &error)) {
        g_print("Error while sending message to '%s':\n%s\n",
                to, error->message);
        ret = false;
      }

      lm_message_unref(m);
      return ret;
    }

    LmHandlerResult LMConnectionC::MessageHandler(LmMessageHandler *handler,
                                                  LmConnection *connection,
                                                  LmMessage *m)
    {
      LmMessageNode *body_node;
      ONDEBUG(std::cerr << "Got message! Type:" << lm_message_get_sub_type(m) << "  \n");
      if(m_dumpRaw) {
        std::cerr << "LMConnection Message: " << lm_message_node_to_string(m->node) << "\n";
      }
      if((lm_message_get_sub_type(m) != LM_MESSAGE_SUB_TYPE_CHAT &&
              lm_message_get_sub_type(m) != LM_MESSAGE_SUB_TYPE_NORMAL)) {
        return LM_HANDLER_RESULT_ALLOW_MORE_HANDLERS;
      }

      body_node = lm_message_node_get_child(m->node, "body");
      if(!body_node) {
        ONDEBUG(std::cerr << "No body for message. \n");
        return LM_HANDLER_RESULT_ALLOW_MORE_HANDLERS;
      }
      m_sigTextMessage(lm_message_node_get_attribute(m->node, "from"), lm_message_node_get_value(body_node));

      return LM_HANDLER_RESULT_REMOVE_MESSAGE;
    }

    //! Called when connection open complete.

    void LMConnectionC::CBConnectionOpen(LmConnection *connection, gboolean success)
    {
      GError *error = NULL;
      ONDEBUG(std::cerr << "Got connection open. \n");
      RavlAssert(connection == m_conn);
      if(!success) {
        std::cerr << "Failed to open a connection. \n";
        return;
      }
      ONDEBUG(std::cerr << "Requesting authentication. User:" << m_user << " Resource:" << m_resource << " Password:" << m_password << "\n");

      if(!lm_connection_authenticate(m_conn,
                                     m_user.data(),
                                     m_password.data(),
                                     m_resource.data(),
                                     &InternalHandlerConnectionAuth, this, FALSE, &error)) {
        std::cerr << "Failed to start authentication.  Message:" << error->message << " \n";
        return;
      }

    }

    void LMConnectionC::CBConnectionAuth(LmConnection *connection, gboolean success)
    {
      ONDEBUG(std::cerr << "Got connection authentication. \n");
      GError *error = NULL;
      RavlAssert(connection == m_conn);
      if(!success) {
        std::cerr << "Failed to authenticate the connection. \n";
        return;
      }
      // Let server know we want messages.
      LmMessage *m = lm_message_new(NULL, LM_MESSAGE_TYPE_PRESENCE);
      if(!lm_connection_send(m_conn, m, &error)) {
        g_error("Connection failed to send presence message: %s",
                error->message);
      }
      lm_message_unref(m);
      m_isReady = true;
    }


    XMLFactoryRegisterC<LMConnectionC> g_xmlFactoryRegister("RavlN::XMPPN::LMConnectionC");

    void LinkRavlXMPPLMConnection()
    {
    }

  }
}
